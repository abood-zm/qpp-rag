import pyterrier as pt
import pyterrier_rag
import pandas as pd
import argparse
import json
import torch
import numpy as np
import pathlib
from tqdm import tqdm
from qpp_methods import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class DocumentContextSelector:
    def __init__(self, strategy="full", encoder_model="sentence-transformers/all-MiniLM-L6-v2", max_docs=10):
        self.strategy = strategy
        self.encoder_model = encoder_model
        self.max_docs = max_docs
        self.tokenizer = None
        self.model = None

        if strategy in ["query_specifity"]:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.model = AutoModel.from_pretrained(encoder_model)
    
    def encode_text(self, text):
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()

    def select_documents(self, query, documents):
        """
        Select which documents to include based on the strategy.

        """
        if not documents:
            return documents
            
        if self.strategy == "full":
            return documents[:self.max_docs]
        elif self.strategy == "none":
            return []
        elif self.strategy == "first_only":
            return documents[:1] if documents else []
        elif self.strategy == "last_only":
            return documents[-1:] if documents else []
        elif self.strategy == "half":
            mid_point = len(documents) // 2
            return documents[:mid_point] if documents else []
        elif self.strategy == "query_specificity":
            return DocumentSpecificity(documents)
        elif self.strategy.startswith("fixed_"): # e.g., fixed_3 for first 3 docs
            num_docs = int(self.strategy.split('_')[1])
            return documents[:num_docs] if documents else []
        elif self.strategy.startswith("random_"): # e.g., random_3 for 3 random docs
            num_docs = int(self.strategy.split('_')[1])
            if len(documents) <= num_docs:
                return documents
            indices = np.random.choice(len(documents), num_docs, replace=False)
            return [documents[i] for i in sorted(indices)]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

class DocumentSpecificity:
    """
    Document specificity would select documents and feed them back into the retrieval based on the following:
    1. Entity Density: Documents with more names, dates, numbers, etc are considered more specific.
    2. Vocabulary Complexity: Documents that have longer words would be more complex.
    3. Document Structure: Documents with lists, citations, and references.
    4. Content Depth: Document with detail indicators and research language.
    5. Query Alignment: TF-IDF (IDF????) similarity focuseding on query-specific terms.
    """
    def __init__(self, documents):
        self.documents = documents
    
    
class DocumentFilteringTransformer(pt.Transformer):
    """
    PyTerrier transformer that filters retrieved documents based on strategy
    """
    def __init__(self, strategy="full", max_docs=10, encoder_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.context_selector = DocumentContextSelector(strategy, encoder_model, max_docs)
        self.logs = []
    
    def transform(self, inp):
        results = []
        
        for qid in inp.qid.unique():
            query_data = inp[inp.qid == qid]
            query = query_data.iloc[0]['query']
            
            # Convert retrieval results to document format
            documents = []
            for _, row in query_data.iterrows():
                doc = {
                    'docno': row.get('docno', ''),
                    'text': row.get('text', ''),
                    'title': row.get('title', ''),
                    'score': row.get('score', 0.0),
                    'rank': row.get('rank', 0)
                }
                documents.append(doc)
            
            # Apply document selection
            selected_docs = self.context_selector.select_documents(query, documents)
            
            # Log selection info
            self.logs.append({
                'qid': qid,
                'query': query,
                'original_docs': len(documents),
                'selected_docs': len(selected_docs),
                'strategy': self.context_selector.strategy
            })
            
            # Convert back to DataFrame format
            for doc in selected_docs:
                results.append({
                    'qid': qid,
                    'query': query,
                    'docno': doc['docno'],
                    'text': doc['text'],
                    'title': doc['title'],
                    'score': doc['score'],
                    'rank': doc.get('rank', 0)
                })
        
        return pd.DataFrame(results) if results else pd.DataFrame(columns=['qid', 'query', 'docno', 'text', 'title', 'score', 'rank'])


def run_document_filtering_experiment(retriever, queries, answers, strategies, k=3, task="nq_test", model="r1", ret="bm25"):
    """
    Run experiments with different document filtering strategies
    """
    results = {}

    for strategy in strategies:
        print(f"Running experiment with strategy: {strategy}")
        
        # Create document filter
        doc_filter = DocumentFilteringTransformer(strategy=strategy, max_docs=k*2)
        
        # Create pipeline: retriever -> document filter -> RAG model -> logging
        filtered_retriever = retriever >> doc_filter
        
        if model == "r1":
            rag_model = pyterrier_rag.SearchR1(filtered_retriever, retrieval_top_k=k)
        elif model == "r1s":
            model_kwargs = {
                'tensor_parallel_size': 1,
                'dtype': 'bfloat16',
                'quantization': 'bitsandbytes',
                'gpu_memory_utilization': 0.7,
                'max_model_len': 92000
            }
            rag_model = pyterrier_rag.R1Searcher(
                filtered_retriever, top_k=k, verbose=False,
                model_kw_args=model_kwargs
            )
        
        # Add logging to the pipeline
        pipeline = rag_model >> pt.apply.generic(lambda x: log_fn(x, ret, answers, k, task, model, strategy=strategy, experiment="filtered", log_type="rag_answers"))
        
        strategy_results = []
        batch_size = 10
        
        for i in tqdm(range(0, len(queries), batch_size), desc=f"Processing {strategy}"):
            batch_queries = queries.iloc[i:i+batch_size]
            # Process the batch through the pipeline
            filtered_df = doc_filter.transform(batch_queries)
            # Fixed: Add log_type parameter for retrieval logging
            log_qpp(filtered_df, ret, k, _task=task, _model=model, strategy=strategy, experiment="filtered", log_type="retrieval")
            batch_results = pipeline(batch_queries)
            # Convert to list of dictionaries for consistency
            for _, row in batch_results.iterrows():
                strategy_results.append({
                    "qid": row['qid'],
                    "query": row['query'],
                    "qanswer": row['qanswer'],
                    "strategy": strategy
                })

        strategy_df = pd.DataFrame(strategy_results)
        eval_results = evaluator(strategy_df, answers)
        
        results[strategy] = {
            "results": strategy_df,
            "evaluation": eval_results,
            "mean_em": eval_results['em'].mean(),
            "mean_f1": eval_results['f1'].mean(),
            "filter_logs": doc_filter.logs
        }
        
        # Save filter logs
        log_filename = f"filtered_{strategy}_{k}_{task}_{model}_logs.json"
        with open(f"./filtered_logs/{log_filename}", 'w') as f:
            json.dump(doc_filter.logs, f, indent=2)
        
        print(f"Strategy {strategy}: EM={results[strategy]['mean_em']:.3f}, F1={results[strategy]['mean_f1']:.3f}")
    
    return results


def evaluator(res, _ans):
    df_content = []
    
    for qid in res.qid.unique():
        golden_answers = _ans[_ans.qid == qid].gold_answer.to_list()
        
        isnull_indicator = res[res.qid==qid].qanswer.isnull().values[0]
        if(isnull_indicator):
            df_content.append([qid, 0.0, 0.0])
            continue
            
        prediction = res[res.qid==qid].qanswer.values[0]
        em_score = pyterrier_rag._measures.ems(prediction, golden_answers)
        f1_list = []
        for a in golden_answers:
            f1_list.append(pyterrier_rag._measures.f1_score(prediction, a))
        f1_score = max(f1_list)
        
        df_content.append([qid, em_score, f1_score])
    
    return pd.DataFrame(df_content, columns=['qid', 'em', 'f1'])


def log_fn(res, _ret, _ans, _k=3, _task='nq_test', _model='r1', experiment=None, strategy=None, log_type=None):
    # Determine output directories based on experiment type
    if experiment == "filtered":
        res_dir = "./filtered_res"
        eval_dir = "./filtered_eval"
    else:
        res_dir = "./res"
        eval_dir = "./eval_res"
    pathlib.Path(res_dir).mkdir(exist_ok=True)
    pathlib.Path(eval_dir).mkdir(exist_ok=True)

    if(_task=='nq_test'):
        if strategy is not None:
            output_filename = f"{_ret}_{_k}_{strategy}_filtered_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{strategy}_{_model}.res"
        else:
            output_filename = f"{_ret}_{_k}_filtered_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{_model}.res"
    else:
        if strategy is not None:
            output_filename = f"{_ret}_{_k}_{strategy}_filtered_{_task}_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{strategy}_{_task}_{_model}.res"
        else:
            output_filename = f"{_ret}_{_k}_filtered_{_task}_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{_task}_{_model}.res"
    
    csvfile = pathlib.Path(f"{res_dir}/{output_filename}")
    eval_csvfile = pathlib.Path(f"{eval_dir}/{output_filename}")
    res.to_csv(f"{res_dir}/{output_filename}", mode='a', index=False, header=not csvfile.exists())
    eval_res = evaluator(res, _ans)
    eval_res.to_csv(f"{eval_dir}/{output_filename}", mode='a', index=False, header=not eval_csvfile.exists())
    return res


def log_qpp(res, _ret, _k, _index=-1, q_encoder=-1, _task='nq_test', _model='r1', strategy=None, experiment=None, log_type=None):
    # Determine output directories based on experiment type
    if experiment == "filtered":
        qpp_dir = "./filtered_qpp_res"
        retrieval_dir = "./filtered_retrieval_res"
    else:
        qpp_dir = "./qpp_res"
        retrieval_dir = "./retrieval_res"
    pathlib.Path(qpp_dir).mkdir(exist_ok=True)
    pathlib.Path(retrieval_dir).mkdir(exist_ok=True)

    # Compose output filename
    if(_task=='nq_test'):
        if strategy is not None:
            if experiment == "filtered" and log_type:
                output_filename = f"{_ret}_{_k}_{strategy}_filtered_{log_type}_{_model}.res"
            elif strategy is not None:
                output_filename = f"{_ret}_{_k}_{strategy}_filtered_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{strategy}_{_model}.res"
            else:
                output_filename = f"{_ret}_{_k}_filtered_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{_model}.res"
        else:
            output_filename = f"{_ret}_{_k}_filtered_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{_model}.res"
    else:
        if strategy is not None:
            if experiment == "filtered" and log_type:
                output_filename = f"{_ret}_{_k}_{strategy}_filtered_{log_type}_{_task}_{_model}.res"
            elif strategy is not None:
                output_filename = f"{_ret}_{_k}_{strategy}_filtered_{_task}_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{strategy}_{_task}_{_model}.res"
            else:
                output_filename = f"{_ret}_{_k}_filtered_{_task}_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{_task}_{_model}.res"
        else:
            output_filename = f"{_ret}_{_k}_filtered_{_task}_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{_task}_{_model}.res"

    csvfile = pathlib.Path(f"{qpp_dir}/{output_filename}")
    retrieval_res_csvfile = pathlib.Path(f"{retrieval_dir}/{output_filename}")
    qpp_df_values = []
    for qid in res.qid.unique():
        sorted_res = res[res.qid==qid].sort_values(by=['score'], ascending=False)
        sorted_res.iloc[:200].to_csv(f"{retrieval_dir}/{output_filename}", mode='a', index=False, header=not retrieval_res_csvfile.exists())
        query_text = res[res.qid==qid]['query'].values[0]
        nqc_est = qpp.nqc(res[res.qid==qid], qid, k=100)
        qpp_df_values.append([qid, query_text, 'nqc', nqc_est, str({'k': 100})])
        if(_index != -1):
            a_ratio_est = qpp.a_ratio_prediction(res[res.qid==qid], qid, _index)
            qpp_df_values.append([qid, query_text, 'a_ratio', a_ratio_est, str({'k': 50, 's1': 0.1, 's2': 0.2})])
            spatial_est = qpp.spatial_prediction(res[res.qid==qid], qid, max(3, _k), q_encoder, _index)
            qpp_df_values.append([qid, query_text, 'spatial', spatial_est, str({'k': max(3, _k)})])
    qpp_df = pd.DataFrame(qpp_df_values, columns=['qid', 'query', 'qpp_method', 'qpp_estimation', 'qpp_parameters']) 
    qpp_df.to_csv(f"{qpp_dir}/{output_filename}", mode='a', index=False, header=not csvfile.exists())
    return res


def load_retriever(_ret, _task='nq_test', _model='r1', experiment="original", strategy=None, k=3):
    if(not pt.java.started()):
        pt.java.init()
    print(f'Loading retriever {_ret} for {_task}....')
    
    if(_task=='nq_test'):
        artifact = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
        sparse_index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
        bm25 = sparse_index.bm25(include_fields=['docno', 'text', 'title'], threads=5)
        bm25_pipeline = pt.rewrite.tokenise() >> bm25 >> pt.rewrite.reset()
    elif(_task=='hotpotqa_dev'):
        sparse_index_path = '../get_res/hotpotqa_sparse_index'
        index_ref = pt.IndexRef.of(sparse_index_path)
        sparse_index = pt.IndexFactory.of(index_ref)
        bm25 = pt.terrier.Retriever(sparse_index, wmodel='BM25')
        bm25_pipeline = pt.rewrite.tokenise() >> bm25 >> sparse_index.text_loader(["text", "title"]) >> pt.rewrite.reset() 

    if(_ret == 'bm25'):
        return bm25_pipeline >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, _task=_task, _model=_model, strategy=strategy, experiment=experiment))
    elif(_ret == 'monoT5'):
        from pyterrier_t5 import MonoT5ReRanker
        monoT5 = MonoT5ReRanker(batch_size=64, verbose=False)
        return (bm25_pipeline % 20) >> monoT5 >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, _task=_task, _model=_model, strategy=strategy, experiment=experiment))
    elif(_ret =='E5'):
        from pyterrier_dr import E5
        import pyterrier_dr
        
        if(_task=='nq_test'):
            e5_index = pt.Artifact.from_hf('pyterrier/ragwiki-e5.flex')
        elif(_task=='hotpotqa_dev'):
            e5_index = pyterrier_dr.FlexIndex('../get_res/e5_hotpotqa_wiki_index_2.flex')

        e5_query_encoder = E5()
        e5_ret = e5_query_encoder >> e5_index.torch_retriever(fp16=True, num_results=120) >> sparse_index.text_loader(["text", "title"])
        return e5_ret >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, e5_index, e5_query_encoder, _task=_task, _model=_model, strategy=strategy, experiment=experiment))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'monoT5', 'E5'])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", type=str, default='r1', choices=['r1', 'r1s'])
    parser.add_argument("--task", type=str, default='nq_test', choices=['nq_test', 'hotpotqa_dev'])
    parser.add_argument("--experiment", type=str, default="original", choices=['original', 'filtered'])
    parser.add_argument("--strategies", type=str, nargs='+', 
                       default=['full', 'none', 'half', 'similarity', 'diversity', 'hybrid'],
                       help="Document selection strategies to test")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of samples to use from the dataset")
    
    args = parser.parse_args()
    k = args.k
    model = args.model
    ret = args.retriever
    task = args.task
    experiment = args.experiment
    strategies = args.strategies
    subset_size = args.subset_size

    # Create directories if they don't exist
    pathlib.Path("./res").mkdir(exist_ok=True)
    pathlib.Path("./eval_res").mkdir(exist_ok=True)
    pathlib.Path("./qpp_res").mkdir(exist_ok=True)
    pathlib.Path("./retrieval_res").mkdir(exist_ok=True)
    
    pathlib.Path("./filtered_res").mkdir(exist_ok=True)
    pathlib.Path("./filtered_eval").mkdir(exist_ok=True)
    pathlib.Path("./filtered_logs").mkdir(exist_ok=True)
    pathlib.Path("./filtered_qpp_res").mkdir(exist_ok=True)
    pathlib.Path("./filtered_retrieval_res").mkdir(exist_ok=True)
    
    # Load datasets
    if(task=='nq_test'):
        queries = pt.get_dataset('rag:nq').get_topics('test')
        answers = pt.get_dataset('rag:nq').get_answers('test')
    elif(task=='hotpotqa_dev'):
        queries = pd.read_csv('./hotpotqa_materials/hotpotqa_queries.csv')
        answers = pd.read_csv('./hotpotqa_materials/hotpotqa_answers.csv')

    # Subset the data if requested
    if subset_size is not None:
        print(f"Using subset of {subset_size} samples from the dataset")
        queries = queries.head(subset_size)
        subset_qids = queries['qid'].tolist()
        answers = answers[answers['qid'].isin(subset_qids)]
        print(f"Subset created: {len(queries)} queries, {len(answers)} answers")
    else:
        print(f"Using full dataset: {len(queries)} queries, {len(answers)} answers")

    qpp = QPP()
    
    if experiment == "original":
        print("Running the original pipeline...")
        retriever = load_retriever(ret, task, model, experiment)
        
        if(model=='r1'):
            print('Loading R1 pipeline ....')
            r1_pipeline = pyterrier_rag.SearchR1(retriever, retrieval_top_k=k) >> pt.apply.generic(lambda x : log_fn(x, ret, answers, k, task, model, experiment))
            print('R1 pipeline is now loaded!')
        elif(model=='r1s'):
            print('Loading R1-Searcher pipeline ....')
            r1_pipeline = pyterrier_rag.R1Searcher(retriever, top_k=k, verbose=False, model_kw_args={'tensor_parallel_size':1, 'dtype':'bfloat16', 'quantization':"bitsandbytes", 'gpu_memory_utilization':0.6, 'max_model_len':92000}) >> pt.apply.generic(lambda x : log_fn(x, ret, answers, k, task, model, experiment))
            print('R1-Searcher pipeline is now loaded!')

        if(task=='nq_test'):
            output_filename = f"{ret}_{k}_{model}.res"
        else:
            output_filename = f"{ret}_{k}_{task}_{model}.res"

        try:
            existing_res_df = pd.read_csv(f"./res/{output_filename}")
            num_existing_qids = existing_res_df.shape[0]
            if subset_size is not None:
                num_existing_qids = 0
                print("Starting fresh due to subset usage")
        except:
            num_existing_qids = 0

        print(f'Retrieval pipeline {ret} for {task} is now loaded!')

        _batch_size = 10
        print(f'Start generation! k={k}, processing {len(queries)} queries')
        for i in tqdm(range(num_existing_qids, len(queries), _batch_size)):
            _batch_queries = queries.iloc[i: i+_batch_size]
            r1_pipeline(_batch_queries)
    
    elif experiment == "filtered":
        print("Running the document filtering experiment...")
        
        base_retriever = load_retriever(ret, task, model, experiment="original")
        
        results = run_document_filtering_experiment(
            retriever=base_retriever, 
            queries=queries, 
            answers=answers, 
            strategies=strategies, 
            k=k, 
            task=task, 
            model=model,
            ret=ret
        )
        
        # Print summary
        print("\n=== DOCUMENT FILTERING EXPERIMENT SUMMARY ===")
        for strategy, data in results.items():
            print(f"{strategy:15s}: EM={data['mean_em']:.3f}, F1={data['mean_f1']:.3f}")