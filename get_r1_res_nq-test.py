import pyterrier as pt
import pyterrier_rag
import pandas as pd
import argparse
import pathlib
from tqdm import tqdm
from qpp_methods import *

def create_standard_rag_pipeline(retriever, k=3, task='nq_test', model='standard_rag'):
    """Create a standard RAG pipeline: retrieve + generate (no reasoning)"""
    
    # Simple generator function for standard RAG
    def standard_rag_generator(df):
        from openai import OpenAI
        import os
        
        client = OpenAI(base_url="http://api.terrier.org/v1", api_key=os.environ["IDA_LLM_API_KEY"])
        
        results = []
        for qid in df.qid.unique():
            query_data = df[df.qid == qid]
            query = query_data.iloc[0]['query']
            
            # Get top-k retrieved documents
            contexts = []
            for _, row in query_data.head(k).iterrows():
                contexts.append(f"Title: {row.get('title', '')}\nText: {row['text']}")
            
            # Build standard RAG prompt
            context_text = "\n\n".join(contexts)
            prompt = f"""Based on the following context, answer the question directly and concisely.

Context:
{context_text}

Question: {query}
Answer:"""
            
            try:
                response = client.completions.create(
                    model='qwen-2.5-7b-instruct',
                    prompt=prompt
                )
                answer = response.choices[0].text.strip() if response.choices else "No Response"
                # Clean the answer
                answer = re.sub(r'^(The answer is|Answer:)\s*', '', answer, flags=re.IGNORECASE)
                answer = answer.strip()
            except Exception as e:
                print(f"Error generating answer for {qid}: {e}")
                answer = "Error generating answer"
            
            results.append({
                'qid': qid,
                'query': query,
                'qanswer': answer,
                'output': answer,
                'iteration': 1,
                'all_queries': str([query])
            })
        
        return pd.DataFrame(results)
    
    # Return the pipeline: retrieve -> generate
    return retriever >> pt.apply.generic(standard_rag_generator) >> pt.apply.generic(lambda x: log_fn(x, 'standard_rag', answers, k, task, model))


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

def log_fn(res, _ret, _ans, _k=3, _task='nq_test', _model='r1'):
    if(_task=='nq_test'):
        output_filename = f"{_ret}_{_k}_{_model}.res"
    else:
        output_filename = f"{_ret}_{_k}_{_task}_{_model}.res"
    csvfile = pathlib.Path(f"./res/{output_filename}")
    eval_csvfile = pathlib.Path(f"./eval_res/{output_filename}")
    res.to_csv(f"./res/{output_filename}", mode='a', index=False, header=not csvfile.exists())
    eval_res = evaluator(res, _ans)
    eval_res.to_csv(f"./eval_res/{output_filename}", mode='a', index=False, header=not eval_csvfile.exists())
    return res

def log_qpp(res, _ret, _k, _index=-1, q_encoder=-1, _task='nq_test', _model='r1'):
    if(_task=='nq_test'):
        output_filename = f"{_ret}_{_k}_{_model}.res"
    else:
        output_filename = f"{_ret}_{_k}_{_task}_{_model}.res"
    csvfile = pathlib.Path(f"./qpp_res/{output_filename}")
    retrieval_res_csvfile = pathlib.Path(f"./retrieval_res/{output_filename}")
    qpp_df_values = []
    for qid in res.qid.unique():
        sorted_res = res[res.qid==qid].sort_values(by=['score'], ascending=False)
        sorted_res.iloc[:200].to_csv(f"./retrieval_res/{output_filename}", mode='a', index=False, header=not retrieval_res_csvfile.exists())
        query_text = res[res.qid==qid]['query'].values[0]
        nqc_est = qpp.nqc(res[res.qid==qid], qid, k=100)
        qpp_df_values.append([qid, query_text, 'nqc', nqc_est, str({'k': 100})])
        if(_index != -1):
            a_ratio_est = qpp.a_ratio_prediction(res[res.qid==qid], qid, _index)
            qpp_df_values.append([qid, query_text, 'a_ratio', a_ratio_est, str({'k': 50, 's1': 0.1, 's2': 0.2})])
            spatial_est = qpp.spatial_prediction(res[res.qid==qid], qid, max(3, _k), q_encoder, _index)
            qpp_df_values.append([qid, query_text, 'spatial', spatial_est, str({'k': max(3, _k)})])
    qpp_df = pd.DataFrame(qpp_df_values, columns=['qid', 'query', 'qpp_method', 'qpp_estimation', 'qpp_parameters']) 
    qpp_df.to_csv(f"./qpp_res/{output_filename}", mode='a', index=False, header=not csvfile.exists())
    return res

def load_retriever(_ret, _task='nq_test', _model='r1'):
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
        return bm25_pipeline >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, _task=_task, _model=_model))
    elif(_ret == 'monoT5'):
        from pyterrier_t5 import MonoT5ReRanker
        monoT5 = MonoT5ReRanker(batch_size=64, verbose=False)
        return (bm25_pipeline % 20) >> monoT5 >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, _task=_task, _model=_model))
    elif(_ret =='E5'):
        from pyterrier_dr import E5
        import pyterrier_dr
        
        if(_task=='nq_test'):
            e5_index = pt.Artifact.from_hf('pyterrier/ragwiki-e5.flex')
        elif(_task=='hotpotqa_dev'):
            e5_index = pyterrier_dr.FlexIndex('../get_res/e5_hotpotqa_wiki_index_2.flex')

        e5_query_encoder = E5()
        e5_ret = e5_query_encoder >> e5_index.torch_retriever(fp16=True, num_results=120) >> sparse_index.text_loader(["text", "title"])
        return e5_ret >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, e5_index, e5_query_encoder, _task=_task, _model=_model))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'monoT5', 'E5'])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", type=str, default='r1', choices=['r1', 'r1s', 'standard_rag'])
    parser.add_argument("--task", type=str, default='nq_test', choices=['nq_test', 'hotpotqa_dev'])
    
    args = parser.parse_args()
    k = args.k
    model = args.model
    ret = args.retriever
    task = args.task
    
    if(task=='nq_test'):
        queries = pt.get_dataset('rag:nq').get_topics('test')
        answers = pt.get_dataset('rag:nq').get_answers('test')
    elif(task=='hotpotqa_dev'):
        queries = pd.read_csv('./hotpotqa_materials/hotpotqa_queries.csv')
        answers = pd.read_csv('./hotpotqa_materials/hotpotqa_answers.csv')

    qpp = QPP()

    retriever = load_retriever(ret, task, model)
    print(f'Retrieval pipeline {ret} for {task} is now loaded!')

    # # test retrieval
    # ret_result = retriever.search('who got the first nobel prize in physics?')
    # print(ret_result)

    if(model=='r1'):
        print('Loading R1 pipeline ....')
        r1_pipeline = pyterrier_rag.SearchR1(retriever, retrieval_top_k=k) >> pt.apply.generic(lambda x : log_fn(x, ret, answers, k, task, model))
        print('R1 pipeline is now loaded!')
    elif(model=='r1s'):
        print('Loading R1-Searcher pipeline ....')
        r1_pipeline = pyterrier_rag.R1Searcher(retriever, top_k=k, verbose=False, model_kw_args={'tensor_parallel_size':1, 'dtype':'bfloat16', 'quantization':"bitsandbytes", 'gpu_memory_utilization':0.6, 'max_model_len':92000}) >> pt.apply.generic(lambda x : log_fn(x, ret, answers, k, task, model))
        print('R1-Searcher pipeline is now loaded!')
        # , model_kw_args={'tensor_parallel_size':1, 'dtype':'bfloat16', 'quantization':"bitsandbytes", 'gpu_memory_utilization':0.6, 'max_model_len':92000}
    elif(model=='standard_rag'):
        print('Loading Standard RAG pipeline ....')
        r1_pipeline = create_standard_rag_pipeline(retriever, k, task, model)
        print('Standard RAG pipeline is now loaded!')
 

    _batch_size = 10

    # check whether there are existing records
    if(task=='nq_test'):
        output_filename = f"{ret}_{k}_{model}.res"
    else:
        output_filename = f"{ret}_{k}_{task}_{model}.res"

    try:
        existing_res_df = pd.read_csv(f"./res/{output_filename}")
        num_existing_qids = existing_res_df.shape[0]
    except:
        num_existing_qids = 0

    print(f'Start generation! k={k}')
    for i in tqdm(range(num_existing_qids, queries.shape[0], _batch_size)):
        _batch_queries = queries.iloc[i: i+_batch_size]
        r1_pipeline(_batch_queries)
