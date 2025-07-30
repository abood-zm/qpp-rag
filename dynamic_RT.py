import pyterrier as pt
import pyterrier_rag
import pandas as pd
import argparse
import pathlib
import os
import threading
from tqdm import tqdm
from qpp_methods import *

class QPPMonitor:
    def __init__(self, qpp_filepath="./qpp_res/bm25_3_r1.res", threshold=0.25, qpp_methods='nqc', min_steps=2, max_steps=10, check_interval = 0.1):
        self.qpp_filepath = qpp_filepath
        self.threshold = threshold
        self.qpp_history = {}
        self.cutoff_decision = {}
        self.min_step = min_steps
        self.max_step = max_steps
        self.last_file_size = 0
        self.check_interval = check_interval
        
    def get_base_qid(self, qid):
        return qid.rsplit('-', 1)[0]
    def get_step_number(self, qid):
        try:
            return int(qid.split('-')[-1])
        except:
            return 0
    def cutoff_decision(self, qid):
        if qid in self.cutoff_decision:
            return self.cutoff_decision[qid]
        return False
    def calculate_qpp(self, qid, qpp_scores):
        current_step = len(qpp_scores)
        if current_step < self.min_steps:
            return False
        if current_step >= self.max_steps:
            return True

        current_score = qpp_scores[-1]
        prev_score = qpp_scores[-2]

        if (current_score - prev_score)/prev_score > self.threshold:
            print(f"Stopping reasoning for {base_qid} due to {drop_percentage:.2%} QPP drop")
            return True
        return False

    def update_from_file(self):
        if not os.path.exists(self.qpp_filepath):
            return
        try:
            current_size = os.path.getsize(self.qpp_filepath)
            if current_size <= self.last_file_size:
                return
            df = pd.read_csv(self.qpp_filepath)
            df_filtered = df[df['qpp_method'] == self.qpp_method].copy()
            for full_qid in df_filtered['qid'].unique():
                base_qid = self.get_base_qid(full_qid)
                step_num = self.get_step_number(full_qid)
                qpp_score = df_filtered[df_fileterd['qid'] == full_qid]['qpp_estimation'].iloc[0]
                if base_qid not in self.qpp_history:
                    self.qpp_history[base_qid] = []
                if len(self.qpp_history[base_qid])==step_num:
                    self.qpp_history[base_qid].append(qpp_score)
                    decision = self.calculate_qpp(base_qid, self.qpp_history[base_qid])
                    self.cutoff_decision[base_qid] = decision
            self.last_file_size = current_size
        except Exception as e:
            print(f"Error reading QPP file: {e}")

class R1Pipeline:
    def __init__(self, base_pipeline, qpp_monitor: QPPMonitor, retriever_with_monitor, answers, k, ret, task, model):
        """
        Wrapper for R1 pipeline with dynamic cutoff capability.
        
        Args:
            base_pipeline: Your existing R1 pipeline (without QPP logging)
            qpp_monitor: QPP file monitor instance
            retriever_with_monitor: Retriever with monitor integration
            answers: Answer dataset for evaluation
            k: Top-k parameter
            ret: Retriever name
            task: Task name
            model: Model name
        """
        self.base_pipeline = base_pipeline
        self.qpp_monitor = qpp_monitor
        self.retriever_with_monitor = retriever_with_monitor
        self.answers = answers
        self.k = k
        self.ret = ret
        self.task = task
        self.model = model
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
    def start_monitoring(self):
        def monitor():
            while not self.stop_monitoring.wait(self.qpp_monitor.check_interval):
                self.qpp_monitor.update_from_file()
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        print(" Started QPP monitoring thread")

    def stop_monitoring_thread(self):
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join()
            print("Stopped monitoring Thread")

    def create_step_pipeline(self, step_num: int):
        if self.model == 'r1':
            step_pipeline = pyterrier_rag.SearchR1(self.retriever_with_monitor, retrieval_top_k=self.k) >> \
                           pt.apply.generic(lambda x: log_fn(x, self.ret, self.answers, self.k, self.task, self.model))
        elif self.model == 'r1s':
            step_pipeline = pyterrier_rag.R1Searcher(
                self.retriever_with_monitor, 
                top_k=self.k, 
                verbose=False, 
                model_kw_args={
                    'tensor_parallel_size': 1, 
                    'dtype': 'bfloat16', 
                    'quantization': "bitsandbytes", 
                    'gpu_memory_utilization': 0.6, 
                    'max_model_len': 92000
                }
            ) >> pt.apply.generic(lambda x: log_fn(x, self.ret, self.answers, self.k, self.task, self.model))
        return step_pipeline

    def search_with_dynamic_cutoff(self, queries_batch: pd.DataFrame) -> pd.DataFrame:
        all_results = []
        for _, query_row in queries_batch.iterrows():
            original_qid = query_row['qid']
            base_qid = self.qpp_monitor.get_base_qid(original_qid)
            query_text = query_row['query']
            
            print(f"Starting multi-step reasoning for query: {query_text}")
            
            if base_qid in self.qpp_monitor.qpp_history:
                del self.qpp_monitor.qpp_history[base_qid]
            if base_qid in self.qpp_monitor.cutoff_decision:
                del self.qpp_monitor.cutoff_decisions[base_qid]
            
            current_step = 0
            final_result = None
            
            while current_step < self.qpp_monitor.max_step:
                step_qid = f"{base_qid}-{current_step}"
                
                step_query = query_row.copy()
                step_query['qid'] = step_qid
                step_query_df = step_query.to_frame().T
                
                print(f"Processing step {current_step} (QID: {step_qid})")
                
                step_pipeline = self.create_step_pipeline(current_step)
                
                try:
                    step_result = step_pipeline(step_query_df)
                    final_result = step_result  # Keep the latest result
                    
                    print(f"Completed step {current_step}")
                    
                except Exception as e:
                    print(f"Error in step {current_step}: {e}")
                    break
                
                time.sleep(0.5)
                
                self.qpp_monitor.update_from_file()
                
                if self.qpp_monitor.should_stop_reasoning(base_qid):
                    print(f"Dynamic cutoff triggered for {base_qid} after step {current_step}")
                    break
                
                if current_step >= self.qpp_monitor.min_steps - 1:
                    # Only check for stopping after minimum steps
                    if base_qid in self.qpp_monitor.qpp_history:
                        qpp_scores = self.qpp_monitor.qpp_history[base_qid]
                        if len(qpp_scores) >= 2:
                            # Show current QPP trend
                            current_qpp = qpp_scores[-1]
                            previous_qpp = qpp_scores[-2]
                            trend = ((current_qpp - previous_qpp) / previous_qpp * 100) if previous_qpp != 0 else 0
                            print(f"QPP trend: {previous_qpp:.2f} → {current_qpp:.2f} ({trend:+.1f}%)")
                
                current_step += 1
            
            if final_result is not None:
                all_results.append(final_result)
                
            if base_qid in self.qpp_monitor.qpp_history:
                qpp_scores = self.qpp_monitor.qpp_history[base_qid]
                print(f"Final QPP history for {base_qid}: {[f'{score:.1f}' for score in qpp_scores]}")
                print(f"Completed after {len(qpp_scores)} steps")
            else:
                print(f"No QPP history found for {base_qid}")
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()


def evaluator(res, _ans):
    df_content = []
    for qid in res.qid.unique():
        golden_answers = _ans[_ans.qid == qid].gold_answer.to_list()
        isnull_indicator = res[res.qid == qid].qanswer.isnull().values[0]
        if isnull_indicator:
            df_content.append([qid, 0.0, 0.0])
            continue
        prediction = res[res.qid == qid].qanswer.values[0]
        em_score = pyterrier_rag._measures.ems(prediction, golden_answers)
        f1_list = [pyterrier_rag._measures.f1_score(prediction, a) for a in golden_answers]
        f1_score = max(f1_list)
        df_content.append([qid, em_score, f1_score])
    return pd.DataFrame(df_content, columns=['qid', 'em', 'f1'])
    
def log_fn(res, _ret, _ans, _k=3, _task='nq_test', _model='r1'):
    if _task == 'nq_test':
        output_filename = f"{_ret}_{_k}_{_model}.res"
    else:
        output_filename = f"{_ret}_{_k}_{_task}_{_model}.res"
    
    csvfile = pathlib.Path(f"./res/{output_filename}")
    eval_csvfile = pathlib.Path(f"./eval_res/{output_filename}")
    
    res.to_csv(csvfile, mode='a', index=False, header=not csvfile.exists())
    
    eval_res = evaluator(res, _ans)
    eval_res.to_csv(eval_csvfile, mode='a', index=False, header=not eval_csvfile.exists())
    
    return res

def modified_log_qpp(res, _ret, _k, _index=-1, q_encoder=-1, _task='nq_test', _model='r1', monitor=None):
    """
    Modified version of your log_qpp function that notifies the monitor.
    """
    # Your existing log_qpp logic
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
    
    # Trigger immediate file check if monitor is provided
    if monitor:
        monitor.update_from_file()
    
    return res

def load_retriever_with_step_logging(_ret, _task='nq_test', _model='r1', monitor=None):
    """
    Modified load_retriever function that includes monitor for QPP logging.
    """
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
        return bm25_pipeline >> pt.apply.generic(lambda x : modified_log_qpp(x, _ret, _k, _task=_task, _model=_model, monitor=monitor))
    elif(_ret == 'monoT5'):
        from pyterrier_t5 import MonoT5ReRanker
        monoT5 = MonoT5ReRanker(batch_size=64, verbose=False)
        return (bm25_pipeline % 20) >> monoT5 >> pt.apply.generic(lambda x : modified_log_qpp(x, _ret, _k, _task=_task, _model=_model, monitor=monitor))
    elif(_ret =='E5'):
        from pyterrier_dr import E5
        import pyterrier_dr
        
        if(_task=='nq_test'):
            e5_index = pt.Artifact.from_hf('pyterrier/ragwiki-e5.flex')
        elif(_task=='hotpotqa_dev'):
            e5_index = pyterrier_dr.FlexIndex('../get_res/e5_hotpotqa_wiki_index_2.flex')

        e5_query_encoder = E5()
        e5_ret = e5_query_encoder >> e5_index.torch_retriever(fp16=True, num_results=120) >> sparse_index.text_loader(["text", "title"])
        return e5_ret >> pt.apply.generic(lambda x : modified_log_qpp(x, _ret, _k, e5_index, e5_query_encoder, _task=_task, _model=_model, monitor=monitor))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'monoT5', 'E5'])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", type=str, default='r1', choices=['r1', 'r1s'])
    parser.add_argument("--task", type=str, default='nq_test', choices=['nq_test', 'hotpotqa_dev'])
    parser.add_argument("--drop_threshold", type=float, default=0.25, help="QPP drop threshold for stopping")
    parser.add_argument("--min_steps", type=int, default=2, help="Minimum reasoning steps")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum reasoning steps")
    parser.add_argument("--qpp_method", type=str, default='nqc', choices=['nqc', 'a_ratio', 'spatial'])
    
    args = parser.parse_args()
    k = args.k
    model = args.model
    ret = args.retriever
    task = args.task
    
    # Load data first
    if(task=='nq_test'):
        queries = pt.get_dataset('rag:nq').get_topics('test')
        answers = pt.get_dataset('rag:nq').get_answers('test')
    elif(task=='hotpotqa_dev'):
        queries = pd.read_csv('./hotpotqa_materials/hotpotqa_queries.csv')
        answers = pd.read_csv('./hotpotqa_materials/hotpotqa_answers.csv')

    # Initialize QPP
    qpp = QPP()
    
    # Determine QPP file path
    if task == 'nq_test':
        qpp_filename = f"{ret}_{k}_{model}.res"
    else:
        qpp_filename = f"{ret}_{k}_{task}_{model}.res"
    
    qpp_file_path = f"./qpp_res/{qpp_filename}"
    
    # Initialize QPP monitor
    qpp_monitor = QPPMonitor(
        qpp_filepath=qpp_file_path,
        threshold=args.drop_threshold,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        qpp_methods=args.qpp_method
    )
    
    # Load retriever with monitor integration
    retriever = load_retriever_with_step_logging(ret, task, model, qpp_monitor)
    print(f'Retrieval pipeline {ret} for {task} is now loaded!')
    
    # Create base R1 pipeline (without QPP logging since retriever handles it)
    if(model=='r1'):
        print('Loading R1 base pipeline ....')
        base_r1_pipeline = pyterrier_rag.SearchR1(retriever, retrieval_top_k=k)
        print('R1 base pipeline is now loaded!')
    elif(model=='r1s'):
        print('Loading R1-Searcher base pipeline ....')
        base_r1_pipeline = pyterrier_rag.R1Searcher(retriever, top_k=k, verbose=False, model_kw_args={'tensor_parallel_size':1, 'dtype':'bfloat16', 'quantization':"bitsandbytes", 'gpu_memory_utilization':0.6, 'max_model_len':92000})
        print('R1-Searcher base pipeline is now loaded!')
    
    # Create dynamic cutoff pipeline
    dynamic_pipeline = R1Pipeline(
        base_pipeline=base_r1_pipeline,
        qpp_monitor=qpp_monitor,
        retriever_with_monitor=retriever,
        answers=answers,
        k=k,
        ret=ret,
        task=task,
        model=model
    )
    
    # Start monitoring
    dynamic_pipeline.start_monitoring()
    
    try:
        # Process with dynamic cutoff
        print(f"🚀 Starting processing with dynamic cutoff (threshold: {args.drop_threshold:.1%})")
        
        # Check for existing results
        try:
            existing_res_df = pd.read_csv(f"./res/{qpp_filename}")
            num_existing_qids = existing_res_df.shape[0]
        except:
            num_existing_qids = 0
        
        print(f'Starting from query {num_existing_qids}')
        
        # Process queries with dynamic cutoff
        for i in range(num_existing_qids, min(num_existing_qids + 10, queries.shape[0])):  # Test with 10 queries
            query_batch = queries.iloc[i:i+1]
            print(f"🔄 Processing query {i}: {query_batch.iloc[0]['query']}")
            result = dynamic_pipeline.search_with_dynamic_cutoff(query_batch)
            print(f"✅ Completed query {i+1}")
            
    finally:
        # Stop monitoring
        dynamic_pipeline.stop_monitoring_thread()
        
    print("🎉 Dynamic cutoff processing completed!")