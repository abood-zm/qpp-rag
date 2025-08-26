import pyterrier as pt
import pyterrier_rag
import pandas as pd
import argparse
import pathlib
from tqdm import tqdm
from qpp_methods import *


qpp_scores_cache = {}
previous_nqc = None
skip_current_query = False
drop_threshold = 0.25  # 25% drop threshold

class QPPMonitor:
    def __init__(self, drop_threshold=0.25):
        self.test_groups = {}  # Store previous NQC for each test group
        self.skip_current_query = False
        self.drop_threshold = drop_threshold
        self.query_history = []  # Store history for analysis
        
    def get_test_group(self, qid):
        """Extract test group from query ID (e.g., 'test_1' from 'test_1-0')"""
        if '-' in qid:
            return qid.rsplit('-', 1)[0]  # Get everything before the last dash
        return qid  # Fallback if no dash found
        
    def get_query_index(self, qid):
        """Extract query index from query ID (e.g., '0' from 'test_1-0')"""
        if '-' in qid:
            return qid.rsplit('-', 1)[1]  # Get everything after the last dash
        return "0"  # Fallback
    
    def calculate_nqc_drop(self, qid, current_nqc):
        """Calculate the percentage drop in NQC score within the same test group"""
        test_group = self.get_test_group(qid)
        query_index = self.get_query_index(qid)
        
        # Get previous NQC for this test group
        previous_nqc = self.test_groups.get(test_group, {}).get('previous_nqc', None)
        
        if previous_nqc is None or query_index == '0':
            return 0.0  # No drop for first query in a test group
        
        # Formula: (qpp_i - qpp_i-1) / qpp_i-1
        drop_percentage = (current_nqc - previous_nqc) / previous_nqc
        return drop_percentage, previous_nqc
    
    def should_skip_query(self, qid, current_nqc):
        """Determine if current query should be skipped based on NQC drop"""
        test_group = self.get_test_group(qid)
        query_index = self.get_query_index(qid)
        
        # Calculate drop within the same test group
        drop_result = self.calculate_nqc_drop(qid, current_nqc)
        if isinstance(drop_result, tuple):
            drop_percentage, previous_nqc = drop_result
        else:
            drop_percentage = drop_result
            previous_nqc = None
        
        # Store query info for history
        query_info = {
            'qid': qid,
            'test_group': test_group,
            'query_index': query_index,
            'nqc': current_nqc,
            'previous_nqc': previous_nqc,
            'drop_percentage': drop_percentage,
            'skipped': False
        }
        
        # Check if drop exceeds threshold (negative drop means decrease)
        if drop_percentage <= -self.drop_threshold and query_index != '0':
            print(f"⚠️  Query {qid}: NQC dropped by {abs(drop_percentage)*100:.1f}% "
                  f"({previous_nqc:.4f} → {current_nqc:.4f}) within {test_group}. SKIPPING!")
            query_info['skipped'] = True
            self.query_history.append(query_info)
            return True
        else:
            if previous_nqc is not None and query_index != '0':
                change_direction = "increased" if drop_percentage >= 0 else "decreased"
                print(f"✓ Query {qid}: NQC {change_direction} by {abs(drop_percentage)*100:.1f}% "
                      f"({previous_nqc:.4f} → {current_nqc:.4f}) within {test_group}. Continuing...")
            else:
                print(f"✓ Query {qid}: First query in {test_group}, NQC = {current_nqc:.4f}. Continuing...")
            
            self.query_history.append(query_info)
            return False
    
    def update_previous_nqc(self, qid, nqc):
        """Update the previous NQC for the specific test group"""
        test_group = self.get_test_group(qid)
        
        if test_group not in self.test_groups:
            self.test_groups[test_group] = {}
        
        self.test_groups[test_group]['previous_nqc'] = nqc
    
    def get_statistics(self):
        """Get statistics about skipped queries"""
        total_queries = len(self.query_history)
        skipped_queries = sum(1 for q in self.query_history if q['skipped'])
        return {
            'total_queries_processed': total_queries,
            'queries_skipped': skipped_queries,
            'skip_rate': skipped_queries / total_queries if total_queries > 0 else 0,
            'history': self.query_history
        }

# Global monitor instance
qpp_monitor = QPPMonitor(drop_threshold=0.25)

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

def log_fn_with_skip_check(res, _ret, _ans, _k=3, _task='nq_test', _model='r1'):
    """Modified log_fn that checks for skip flag"""
    # Check if this query should be skipped
    if 'skip_query' in res.columns and res['skip_query'].iloc[0] == True:
        print(f"🚫 Skipping R1 processing for query {res['qid'].iloc[0]} due to QPP drop")
        
        # Create a dummy result to maintain pipeline flow
        skipped_result = res[['qid', 'query']].copy()
        skipped_result['qanswer'] = None  # No answer generated
        skipped_result['skipped'] = True
        
        return skipped_result
    
    # Normal processing for non-skipped queries
    return log_fn(res, _ret, _ans, _k, _task, _model)

def log_qpp_with_monitoring(res, _ret, _k, _index=-1, q_encoder=-1, _task='nq_test', _model='r1'):
    """Modified log_qpp function with real-time monitoring and skipping"""
    global qpp_monitor
    
    # Original file saving logic
    if(_task=='nq_test'):
        output_filename = f"{_ret}_{_k}_{_model}.res"
    else:
        output_filename = f"{_ret}_{_k}_{_task}_{_model}.res"
    
    csvfile = pathlib.Path(f"./qpp_res/{output_filename}")
    retrieval_res_csvfile = pathlib.Path(f"./retrieval_res/{output_filename}")
    qpp_df_values = []
    
    for qid in res.qid.unique():
        sorted_res = res[res.qid==qid].sort_values(by=['score'], ascending=False)
        sorted_res.iloc[:200].to_csv(f"./retrieval_res/{output_filename}", mode='a', index=False, 
                                    header=not retrieval_res_csvfile.exists())
        
        query_text = res[res.qid==qid]['query'].values[0]
        
        # Calculate NQC score
        nqc_est = qpp.nqc(res[res.qid==qid], qid, k=100)
        
        # Real-time monitoring and decision making
        should_skip = qpp_monitor.should_skip_query(qid, nqc_est)
        
        if should_skip:
            # Mark this query to be skipped
            # You can either return early or set a flag
            qpp_monitor.update_previous_nqc(qid, nqc_est)  # Update for next iteration
            
            # Still log the QPP data but mark it as skipped
            qpp_df_values.append([qid, query_text, 'nqc', nqc_est, str({'k': 100, 'skipped': True})])
            
            # Create a modified result that signals to skip this query
            skip_res = res[res.qid==qid].copy()
            skip_res['skip_query'] = True
            return skip_res
        
        # Continue with normal processing for non-skipped queries
        qpp_monitor.update_previous_nqc(qid, nqc_est)
        
        # Original QPP calculations
        qpp_df_values.append([qid, query_text, 'nqc', nqc_est, str({'k': 100})])
        
        if(_index != -1):
            a_ratio_est = qpp.a_ratio_prediction(res[res.qid==qid], qid, _index)
            qpp_df_values.append([qid, query_text, 'a_ratio', a_ratio_est, 
                                str({'k': 50, 's1': 0.1, 's2': 0.2})])
            spatial_est = qpp.spatial_prediction(res[res.qid==qid], qid, max(3, _k), q_encoder, _index)
            qpp_df_values.append([qid, query_text, 'spatial', spatial_est, str({'k': max(3, _k)})])
    
    # Save QPP results
    qpp_df = pd.DataFrame(qpp_df_values, columns=['qid', 'query', 'qpp_method', 'qpp_estimation', 'qpp_parameters']) 
    qpp_df.to_csv(f"./qpp_res/{output_filename}", mode='a', index=False, header=not csvfile.exists())
    
    return res
    
def get_qpp_monitoring_stats():
    """Get current monitoring statistics"""
    return qpp_monitor.get_statistics()
    
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
        return bm25_pipeline >> pt.apply.generic(lambda x : log_qpp_with_monitoring(x, _ret, k, _task=_task, _model=_model))
    elif(_ret == 'monoT5'):
        from pyterrier_t5 import MonoT5ReRanker
        monoT5 = MonoT5ReRanker(batch_size=64, verbose=False)
        return (bm25_pipeline % 20) >> monoT5 >> pt.apply.generic(lambda x : log_qpp_with_monitoring(x, _ret, k, _task=_task, _model=_model))
    elif(_ret =='E5'):
        from pyterrier_dr import E5
        import pyterrier_dr
        
        if(_task=='nq_test'):
            e5_index = pt.Artifact.from_hf('pyterrier/ragwiki-e5.flex')
        elif(_task=='hotpotqa_dev'):
            e5_index = pyterrier_dr.FlexIndex('../get_res/e5_hotpotqa_wiki_index_2.flex')

        e5_query_encoder = E5()
        e5_ret = e5_query_encoder >> e5_index.torch_retriever(fp16=True, num_results=120) >> sparse_index.text_loader(["text", "title"])
        return e5_ret >> pt.apply.generic(lambda x : log_qpp_with_monitoring(x, _ret, k, e5_index, e5_query_encoder, _task=_task, _model=_model))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'monoT5', 'E5'])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", type=str, default='r1', choices=['r1', 'r1s'])
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
        print('Loading R1 pipeline with QPP monitoring....')
        r1_pipeline = pyterrier_rag.SearchR1(retriever, retrieval_top_k=k) >> pt.apply.generic(lambda x : log_fn_with_skip_check(x, ret, answers, k, task, model))
        print('R1 pipeline with monitoring is now loaded!')

    elif(model=='r1s'):
        print('Loading R1-Searcher pipeline ....')
        r1_pipeline = pyterrier_rag.R1Searcher(retriever, top_k=k, verbose=False, model_kw_args={'tensor_parallel_size':1, 'dtype':'bfloat16', 'quantization':"bitsandbytes", 'gpu_memory_utilization':0.6, 'max_model_len':92000}) >> pt.apply.generic(lambda x : log_fn(x, ret, answers, k, task, model))
        print('R1-Searcher pipeline is now loaded!')
        # , model_kw_args={'tensor_parallel_size':1, 'dtype':'bfloat16', 'quantization':"bitsandbytes", 'gpu_memory_utilization':0.6, 'max_model_len':92000}

    ## test r1 pipeline
    # r1_result = r1_pipeline.search("What are chemical reactions?")
    # print(r1_result)

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

    print("\n📊 QPP Monitoring Statistics:")
    stats = get_qpp_monitoring_stats()
    print(f"Total queries processed: {stats['total_queries_processed']}")
    print(f"Queries skipped: {stats['queries_skipped']}")
    print(f"Skip rate: {stats['skip_rate']:.2%}")
