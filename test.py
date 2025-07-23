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

        if strategy in ["similarity", 'diversity', 'hybrid']:
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
        
        Args:
            query: The current query string
            documents: List of document dictionaries with 'text' and other fields
            
        Returns:
            List of selected documents
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
        elif self.strategy == "similarity":
            return self._similarity_selection(query, documents)
        elif self.strategy == "diversity":
            return self._diversity_selection(documents)
        elif self.strategy == "hybrid":
            return self._hybrid_selection(query, documents)
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

    def _similarity_selection(self, query, documents, top_k=5):
        if not documents:
            return []
        
        query_embedding = self.encode_text(query)
        
        # Extract text from documents for encoding
        doc_texts = []
        for doc in documents:
            # Try different possible text fields
            text = doc.get('text', '') or doc.get('body', '') or doc.get('content', '') or str(doc)
            doc_texts.append(text)
        
        doc_embeddings = self.encode_text(doc_texts)
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        top_k = min(top_k, len(documents))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [documents[i] for i in top_indices]
    
    def _diversity_selection(self, documents, max_docs=5):
        if not documents:
            return []
        
        if len(documents) <= max_docs:
            return documents
        
        # Extract text from documents
        doc_texts = []
        for doc in documents:
            text = doc.get('text', '') or doc.get('body', '') or doc.get('content', '') or str(doc)
            doc_texts.append(text)
        
        embeddings = self.encode_text(doc_texts)
        selected_indices = [0]  # Start with first document

        # Greedy diversity selection
        for _ in range(max_docs - 1):
            if len(selected_indices) >= len(documents):
                break

            best_candidate = -1
            best_min_similarity = -1

            for i in range(len(documents)):
                if i in selected_indices:
                    continue

                # Calculate minimum similarity to already selected documents
                min_sim = min([cosine_similarity(embeddings[i:i+1], embeddings[j:j+1])[0][0] 
                              for j in selected_indices])

                if min_sim > best_min_similarity:
                    best_min_similarity = min_sim
                    best_candidate = i

            if best_candidate != -1:
                selected_indices.append(best_candidate)
        
        return [documents[i] for i in selected_indices]
    
    def _hybrid_selection(self, query, documents):
        if not documents:
            return []
        
        # First get top similar documents
        sim_docs = self._similarity_selection(query, documents, top_k=min(8, len(documents)))
        
        # Then apply diversity selection on the similar documents
        if len(sim_docs) > 3:
            return self._diversity_selection(sim_docs, max_docs=3)
        
        return sim_docs


class DocumentFilteringTransformer(pt.Transformer):
    """
    PyTerrier transformer that filters retrieved documents based on strategy
    """
    def __init__(self, strategy="full", max_docs=10, encoder_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.context_selector = DocumentContextSelector(strategy, encoder_model, max_docs)
        self.logs = []
        self.strategy = strategy  # Store strategy for easy access
    
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


def create_all_directories():
    """Create all necessary directories at startup"""
    directories = [
        "./res", "./eval_res", "./qpp_res", "./retrieval_res",
        "./filtered_res", "./filtered_eval", "./filtered_logs", 
        "./filtered_qpp_res", "./filtered_retrieval_res"
    ]
    for directory in directories:
        pathlib.Path(directory).mkdir(exist_ok=True)
    print("All directories created successfully")


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
        
        # Add logging to the pipeline with proper parameters
        pipeline = rag_model >> pt.apply.generic(
            lambda x: log_fn(x, ret, answers, k, task, model, 
                           experiment="filtered", strategy=strategy)
        )
        
        batch_size = 10
        
        # Process all queries through the pipeline in batches
        for i in tqdm(range(0, len(queries), batch_size), desc=f"Processing {strategy}"):
            batch_queries = queries.iloc[i:i+batch_size]
            try:
                # Process the batch through the pipeline - this will automatically save via log_fn
                pipeline(batch_queries)
            except Exception as e:
                print(f"Error processing batch {i} for strategy {strategy}: {e}")
                continue
        
        # Read the saved results to compute evaluation metrics
        if task == 'nq_test':
            output_filename = f"{ret}_{k}_{strategy}_filtered_{model}.res"
        else:
            output_filename = f"{ret}_{k}_{strategy}_filtered_{task}_{model}.res"
        
        try:
            strategy_df = pd.read_csv(f"./filtered_res/{output_filename}")
            if strategy_df.empty:
                print(f"Warning: Empty results file for strategy {strategy}")
                eval_results = pd.DataFrame(columns=['qid', 'em', 'f1'])
                mean_em = mean_f1 = 0.0
            else:
                eval_results = evaluator(strategy_df, answers)
                mean_em = eval_results['em'].mean()
                mean_f1 = eval_results['f1'].mean()
            
            results[strategy] = {
                "results": strategy_df,
                "evaluation": eval_results,
                "mean_em": mean_em,
                "mean_f1": mean_f1,
                "filter_logs": doc_filter.logs
            }
            
            print(f"Strategy {strategy}: EM={results[strategy]['mean_em']:.3f}, F1={results[strategy]['mean_f1']:.3f}")
            
        except FileNotFoundError:
            print(f"Warning: Could not find results file for strategy {strategy}: ./filtered_res/{output_filename}")
            results[strategy] = {
                "results": pd.DataFrame(),
                "evaluation": pd.DataFrame(),
                "mean_em": 0.0,
                "mean_f1": 0.0,
                "filter_logs": doc_filter.logs
            }
        except Exception as e:
            print(f"Error processing results for strategy {strategy}: {e}")
            results[strategy] = {
                "results": pd.DataFrame(),
                "evaluation": pd.DataFrame(),
                "mean_em": 0.0,
                "mean_f1": 0.0,
                "filter_logs": doc_filter.logs
            }
        
        # Save filter logs
        try:
            log_filename = f"filtered_{strategy}_{k}_{task}_{model}_logs.json"
            with open(f"./filtered_logs/{log_filename}", 'w') as f:
                json.dump(doc_filter.logs, f, indent=2)
            print(f"Saved filter logs to ./filtered_logs/{log_filename}")
        except Exception as e:
            print(f"Error saving filter logs for strategy {strategy}: {e}")
    
    return results


def evaluator(res, _ans):
    """Evaluate predictions against gold answers"""
    df_content = []
    
    for qid in res.qid.unique():
        golden_answers = _ans[_ans.qid == qid].gold_answer.to_list()
        
        if res[res.qid==qid].empty:
            df_content.append([qid, 0.0, 0.0])
            continue
            
        qanswer_vals = res[res.qid==qid].qanswer.values
        if len(qanswer_vals) == 0:
            df_content.append([qid, 0.0, 0.0])
            continue
            
        isnull_indicator = pd.isnull(qanswer_vals[0])
        if isnull_indicator:
            df_content.append([qid, 0.0, 0.0])
            continue
            
        prediction = qanswer_vals[0]
        em_score = pyterrier_rag._measures.ems(prediction, golden_answers)
        f1_list = []
        for a in golden_answers:
            f1_list.append(pyterrier_rag._measures.f1_score(prediction, a))
        f1_score = max(f1_list) if f1_list else 0.0
        
        df_content.append([qid, em_score, f1_score])
    
    return pd.DataFrame(df_content, columns=['qid', 'em', 'f1'])


def log_fn(res, _ret, _ans, _k=3, _task='nq_test', _model='r1', experiment=None, strategy=None):
    """Improved logging function with better error handling and debug info"""
    
    # Determine output directories based on experiment type
    if experiment == "filtered":
        res_dir = "./filtered_res"
        eval_dir = "./filtered_eval"
    else:
        res_dir = "./res"
        eval_dir = "./eval_res"
    
    # Ensure directories exist
    pathlib.Path(res_dir).mkdir(exist_ok=True)
    pathlib.Path(eval_dir).mkdir(exist_ok=True)

    # Generate filename
    if _task == 'nq_test':
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

    # Debug print statements
    print(f"[log_fn] Strategy: {strategy}, Experiment: {experiment}")
    print(f"[log_fn] Logging {res.shape[0]} rows to {res_dir}/{output_filename}")
    print(f"[log_fn] Columns: {res.columns.tolist()}")
    
    # Check if res has required columns
    required_cols = ['qid', 'query']
    missing_cols = [col for col in required_cols if col not in res.columns]
    if missing_cols:
        print(f"[log_fn] WARNING: Missing required columns: {missing_cols}")
    
    # Check for qanswer column specifically
    if 'qanswer' not in res.columns:
        print(f"[log_fn] WARNING: 'qanswer' column not found. Available columns: {res.columns.tolist()}")

    try:
        # Save main results
        res.to_csv(f"{res_dir}/{output_filename}", mode='a', index=False, header=not csvfile.exists())
        print(f"[log_fn] Successfully saved results to {res_dir}/{output_filename}")
        
        # Save evaluation results
        eval_res = evaluator(res, _ans)
        eval_res.to_csv(f"{eval_dir}/{output_filename}", mode='a', index=False, header=not eval_csvfile.exists())
        print(f"[log_fn] Successfully saved evaluation to {eval_dir}/{output_filename}")
        
    except Exception as e:
        print(f"[log_fn] ERROR saving files: {e}")
        print(f"[log_fn] Result shape: {res.shape}")
        print(f"[log_fn] Result columns: {res.columns.tolist()}")
        if not res.empty:
            print(f"[log_fn] First few rows:\n{res.head()}")
    
    return res


def log_qpp(res, _ret, _k, _index=-1, q_encoder=-1, _task='nq_test', _model='r1', strategy=None, experiment=None):
    """QPP logging function - only used for original experiment"""
    
    # Skip QPP logging for filtered experiments to avoid conflicts
    if experiment == "filtered":
        return res
        
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
    if _task == 'nq_test':
        if strategy is not None:
            output_filename = f"{_ret}_{_k}_{strategy}_filtered_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{strategy}_{_model}.res"
        else:
            output_filename = f"{_ret}_{_k}_filtered_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{_model}.res"
    else:
        if strategy is not None:
            output_filename = f"{_ret}_{_k}_{strategy}_filtered_{_task}_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{strategy}_{_task}_{_model}.res"
        else:
            output_filename = f"{_ret}_{_k}_filtered_{_task}_{_model}.res" if experiment == "filtered" else f"{_ret}_{_k}_{_task}_{_model}.res"

    csvfile = pathlib.Path(f"{qpp_dir}/{output_filename}")
    retrieval_res_csvfile = pathlib.Path(f"{retrieval_dir}/{output_filename}")
    qpp_df_values = []
    
    try:
        for qid in res.qid.unique():
            sorted_res = res[res.qid==qid].sort_values(by=['score'], ascending=False)
            sorted_res.iloc[:200].to_csv(f"{retrieval_dir}/{output_filename}", mode='a', index=False, header=not retrieval_res_csvfile.exists())
            query_text = res[res.qid==qid]['query'].values[0]
            
            # QPP calculations
            nqc_est = qpp.nqc(res[res.qid==qid], qid, k=100)
            qpp_df_values.append([qid, query_text, 'nqc', nqc_est, str({'k': 100})])
            
            if _index != -1:
                a_ratio_est = qpp.a_ratio_prediction(res[res.qid==qid], qid, _index)
                qpp_df_values.append([qid, query_text, 'a_ratio', a_ratio_est, str({'k': 50, 's1': 0.1, 's2': 0.2})])
                spatial_est = qpp.spatial_prediction(res[res.qid==qid], qid, max(3, _k), q_encoder, _index)
                qpp_df_values.append([qid, query_text, 'spatial', spatial_est, str({'k': max(3, _k)})])
        
        qpp_df = pd.DataFrame(qpp_df_values, columns=['qid', 'query', 'qpp_method', 'qpp_estimation', 'qpp_parameters']) 
        qpp_df.to_csv(f"{qpp_dir}/{output_filename}", mode='a', index=False, header=not csvfile.exists())
        
    except Exception as e:
        print(f"[log_qpp] Error in QPP logging: {e}")
    
    return res


def load_retriever(_ret, _task='nq_test', _model='r1', experiment="original", strategy=None, k=3):
    """Load retriever with appropriate logging based on experiment type"""
    
    if not pt.java.started():
        pt.java.init()
    print(f'Loading retriever {_ret} for {_task}....')
    
    if _task == 'nq_test':
        artifact = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
        sparse_index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
        bm25 = sparse_index.bm25(include_fields=['docno', 'text', 'title'], threads=5)
        bm25_pipeline = pt.rewrite.tokenise() >> bm25 >> pt.rewrite.reset()
    elif _task == 'hotpotqa_dev':
        sparse_index_path = '../get_res/hotpotqa_sparse_index'
        index_ref = pt.IndexRef.of(sparse_index_path)
        sparse_index = pt.IndexFactory.of(index_ref)
        bm25 = pt.terrier.Retriever(sparse_index, wmodel='BM25')
        bm25_pipeline = pt.rewrite.tokenise() >> bm25 >> sparse_index.text_loader(["text", "title"]) >> pt.rewrite.reset() 

    if _ret == 'bm25':
        # Only add QPP logging for original experiment
        if experiment == "original":
            return bm25_pipeline >> pt.apply.generic(lambda x: log_qpp(x, _ret, k, _task=_task, _model=_model, strategy=strategy, experiment=experiment))
        else:
            return bm25_pipeline
            
    elif _ret == 'monoT5':
        from pyterrier_t5 import MonoT5ReRanker
        monoT5 = MonoT5ReRanker(batch_size=64, verbose=False)
        if experiment == "original":
            return (bm25_pipeline % 20) >> monoT5 >> pt.apply.generic(lambda x: log_qpp(x, _ret, k, _task=_task, _model=_model, strategy=strategy, experiment=experiment))
        else:
            return (bm25_pipeline % 20) >> monoT5
            
    elif _ret == 'E5':
        from pyterrier_dr import E5
        import pyterrier_dr
        
        if _task == 'nq_test':
            e5_index = pt.Artifact.from_hf('pyterrier/ragwiki-e5.flex')
        elif _task == 'hotpotqa_dev':
            e5_index = pyterrier_dr.FlexIndex('../get_res/e5_hotpotqa_wiki_index_2.flex')

        e5_query_encoder = E5()
        e5_ret = e5_query_encoder >> e5_index.torch_retriever(fp16=True, num_results=120) >> sparse_index.text_loader(["text", "title"])
        
        if experiment == "original":
            return e5_ret >> pt.apply.generic(lambda x: log_qpp(x, _ret, k, e5_index, e5_query_encoder, _task=_task, _model=_model, strategy=strategy, experiment=experiment))
        else:
            return e5_ret


if __name__ == "__main__":
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

    # Create all directories at startup
    create_all_directories()
    
    # Load datasets
    if task == 'nq_test':
        queries = pt.get_dataset('rag:nq').get_topics('test')
        answers = pt.get_dataset('rag:nq').get_answers('test')
    elif task == 'hotpotqa_dev':
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
        retriever = load_retriever(ret, task, model, experiment, k=k)
        
        if model == 'r1':
            print('Loading R1 pipeline ....')
            r1_pipeline = pyterrier_rag.SearchR1(retriever, retrieval_top_k=k) >> pt.apply.generic(lambda x: log_fn(x, ret, answers, k, task, model, experiment))
            print('R1 pipeline is now loaded!')
        elif model == 'r1s':
            print('Loading R1-Searcher pipeline ....')
            r1_pipeline = pyterrier_rag.R1Searcher(retriever, top_k=k, verbose=False, model_kw_args={'tensor_parallel_size':1, 'dtype':'bfloat16', 'quantization':"bitsandbytes", 'gpu_memory_utilization':0.6, 'max_model_len':92000}) >> pt.apply.generic(lambda x: log_fn(x, ret, answers, k, task, model, experiment))
            print('R1-Searcher pipeline is now loaded!')

        if task == 'nq_test':
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
            try:
                r1_pipeline(_batch_queries)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
    
    elif experiment == "filtered":
        print("Running the document filtering experiment...")
        
        # Load base retriever WITHOUT QPP logging for filtered experiment
        base_retriever = load_retriever(ret, task, model, experiment="filtered", k=k)
        
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
            print(f"{strategy:15s}: EM={data['mean_em']:.3f},