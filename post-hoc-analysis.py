
import pandas as pd
import argparse
import ast
import re
import numpy
import json
import os
from openai import OpenAI
from typing import Dict, List, Any
import logging
from datetime import datetime
import pathlib
import pyterrier as pt
import pyterrier_rag._measures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostAnalysis:
    def __init__(self, res_filepath, qpp_filepath, cutoff, task=None, dynamic=False, ans_filepath=None, output_dir='rag/analysis_results'):
        self.client = OpenAI(base_url="http://api.terrier.org/v1", api_key=os.environ["IDA_LLM_API_KEY"])
        self.result = []
        self.output_dir = output_dir
        self.results_df = None
        self.task = task
        self.qpp_filepath = qpp_filepath
        self.res_filepath = res_filepath
        self.dynamic = dynamic
        self.qpp_df = None
        self.dynamic_cutoff = {}

        if dynamic and qpp_filepath:
            self.qpp_df = pd.read_csv(self.qpp_filepath)
            print("Dynamic Cutoff data has been loaded...")
            self.dynamic_cutoff = self.calculate_dynamic_cutoff()
        else:
            self.cutoff = cutoff
        self.extracted_data = self.extract_intermediate_queries()
    
        self.ans_data = None
        
        if ans_filepath:
            self.ans_data = pd.read_csv(ans_filepath)
        else:
            if task == "nq_test":
                if not pt.java.started():
                    pt.java.init()
                self.ans_data = pt.get_dataset('rag:nq').get_answers('test')
                print(type(self.ans_data))
                logger.info("Loaded NQ test answers from PyTerrier dataset")
            elif task == "hotpot_qa":
                self.ans_data = self.extract_answers(self.extracted_data)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("./res", exist_ok=True)
        os.makedirs("./eval_res", exist_ok=True)

    def extract_answers(self, data):
        answers = []
        for qid, data in data.items():
            answers.append({
                "qid": qid,
                "gold_answer": data['golden_answer']
            })
        return pd.DataFrame(answers)
    def calculate_dynamic_cutoff(self):
        cutoff = {}
        res_df = pd.read_csv(self.res_filepath)

        for _, res in res_df.iterrows():
            qid = res['qid']
            iteration = res['iteration']

            if iteration > 1:
                base_qid = qid.split('-')[0]
                qpp_subset = self.qpp_df[self.qpp_df['qid'].str.startswith(base_qid)]

                def extract_iteration_number(qid_series):
                    return qid_series.str.extract(r"-(\d+)$").fillna("0").astype(int).iloc[:, 0]
                qpp_subset = qpp_subset.sort_values(by='qid', key=extract_iteration_number)
                count = 1
                for i in range(1, min(iteration, len(qpp_subset))):
                    if i >= len(qpp_subset) or (i - 1) >= len(qpp_subset):
                        print("WARNING")
                        break
                    prev = qpp_subset.iloc[i - 1]['qpp_estimation']
                    curr = qpp_subset.iloc[i]['qpp_estimation']

                    if (prev - curr)/prev > 0.25:
                        print(f"QID {qid} has hit the threshold. its count is {count}")
                        break
                    count += 1
                cutoff[qid] = count
            else:
                cutoff[qid] = 1
        return cutoff

    def extract_intermediate_queries(self):
        res_df = pd.read_csv(self.res_filepath)
        extracted_data = {}
        
        for _,res in res_df.iterrows():
            qid = res['qid']
            query = res['query']
            output = res['output']
            golden_answer = res['qanswer']
            
            if "<|im_start|>assistant" in output:
                assistant_output = output.split("<|im_start|>assistant", 1)[1]
                if "<|im_end|>" in assistant_output:
                    assistant_output = assistant_output.split("<|im_end|>")[0]
            else:
                assistant_output = output

            search_pattern = r"(<think>.*?</think>\s*<search>.*?</search>\s*<information>.*?</information>)"
            matches = re.findall(search_pattern, assistant_output, flags=re.DOTALL)
            
            if self.dynamic:
                current_cutoff = self.dynamic_cutoff[qid]
                if current_cutoff == res['iteration']:
                    print(f"for {qid}, dynamic cutoff is same as iteration")
            else:
                current_cutoff = self.cutoff
            
            if current_cutoff == 0:
                intermediate_queries = []
            else:
                intermediate_queries = matches[:current_cutoff]
                
            extracted_data[qid] = {
                    "query": query,
                    "inter_queries": matches[:current_cutoff],
                    "golden_answer": golden_answer,
                    "cutoff_used": current_cutoff,
                    "cutoff_type": "dynamic" if (self.dynamic and qid in self.dynamic_cutoff) else "fixed"
                }
    
        return extracted_data
    
    def build_prompt(self, original_query, intermediate_queries):
        if not intermediate_queries:
            return f"Generate a short answer to this question: {original_query}"
    
        intermediate_content = ""
        for qid, query in enumerate(intermediate_queries):
            
            think_pattern = r"<think>(.*?)</think>"
            think_part = re.search(think_pattern, query, flags=re.DOTALL)
            if not think_part:
                print("DEBUG - Think part is not found!")
                print(f"DEBUG - Full query: {query}")
            think_term = think_part.group(1).strip() if think_part else "N/A"
                
            search_pattern = r"<search>(.*?)</search>"
            search_part = re.search(search_pattern, query, flags=re.DOTALL)
            search_term = search_part.group(1).strip() if search_part else "N/A"
            
            info_pattern = r"<information>(.*?)</information>"
            info_part = re.search(info_pattern, query, flags=re.DOTALL)
            if not info_part:
                print("DEBUG - Info part not found!")
            info_term = info_part.group(1).strip() if info_part else "N/A"
            
            intermediate_content += f"<think> {think_term} </think>\n\n<search> {search_term} </search>\n\n<information> {info_term} </information>"

        prompt4 = f""""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: Were Scott Derrickson and Ed Wood of the same nationality?
<|im_end|>
<|im_start|>assistant

{intermediate_content.strip()}

<answer>
"""
        return prompt4
        
    def generator_model(self, prompt, qid):

        try:
            result = self.client.completions.create(model='qwen-2.5-7b-instruct',prompt=prompt, max_tokens=100)
            raw_answer = result.choices[0].text.strip() if result.choices else "No Response"
            
            # Clean the generated answer
            generated_answer = raw_answer
            generated_answer = re.sub(r'^(The answer is|Answer:)\s*', '', generated_answer, flags=re.IGNORECASE)
            generated_answer = re.sub(r'</?answer>', '', generated_answer)
            generated_answer = generated_answer.strip()

            logger.info(f"Successfully generated answer for QID: {qid}")
            return {
                "generated_answer": generated_answer,
                "raw_answer": raw_answer, 
                "status":"success",
                "error":None
            }

        except Exception as e:
            logger.info(f"The answer was not generated successfully for QID {qid}: {str(e)}")
            return {
                "generated_answer":None,
                "status":"failed",
                "error":str(e)
            }

    def generating_answers(self):
        logger.info(f"Starting analysis with cutoff: {'dynamic' if dynamic else self.cutoff} ")
        logger.info(f"Processing {len(self.extracted_data)} queries")
        for idx, (qid, data) in enumerate(self.extracted_data.items()):
            logger.info(f"Processing QID: {qid}")
            context_parts = []
            for i in range(min(int(data['cutoff_used']), len(data['inter_queries']))):
                prompt = self.build_prompt(data['query'], [data['inter_queries'][i]])
                model_result = self.generator_model(prompt, qid)
                if model_result['generated_answer']:
                    clean_answer = re.sub(r'</?answer>', '', model_result['generated_answer']).strip()
                    context_parts.append(f"Search {i+1}: {clean_answer}")
            
            context = " | ".join(context_parts) if context_parts else "No context available"
            
            final_prompt = f"""<|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Answer the question: {data['query']}
    
    Context from searches: {context}
    
    Provide only the final answer in 1-4 words. Do not explain or add extra text.
    <|im_end|>
    <|im_start|>assistant
    Answer: """
            
            final_result = self.generator_model(final_prompt, qid)
            if final_result['generated_answer']:
                final_answer = re.sub(r'</?answer>', '', final_result['generated_answer']).strip()
                final_answer = re.sub(r'^(The answer is|Answer:)\s*', '', final_answer, flags=re.IGNORECASE)
                if final_answer == "Yes" or final_answer == "No":
                    final_answer.lower()
                final_result['generated_answer'] = final_answer

                result_entry = {
                        "qid": qid,
                        "original_query": data['query'],
                        "golden_answer": data['golden_answer'],
                        "num_intermediate_queries": len(data['inter_queries']),
                        "intermediate_queries": data['inter_queries'],
                        "prompt": prompt,
                        "generated_answer": final_result['generated_answer'],
                        "model_status": final_result['status'],
                        "model_error": final_result['error']
                    }
            self.result.append(result_entry)
            print(f"QID: {qid}")
            print(f"Prompt:{final_prompt}")
            print(f"Original Query: {data['query']}")
            print(f"Golden Answer: '{data['golden_answer']}'")
            print(f"Generated Answer: '{final_result['generated_answer']}'")
            print(f"Generated Answer Length: {len(final_result['generated_answer']) if final_result['generated_answer'] else 0}")
            print(f"Exact Match: {data['golden_answer'] == final_result['generated_answer']}")
            print(f"Intermediate Queries Count: {len(data['inter_queries'])}")
            print("-" * 80)

    def prepare_results(self):
        df_data = []
        for result in self.result:
            df_data.append({
                'qid': result['qid'],
                'query': result['original_query'],
                'qanswer': result['generated_answer'],
                'output': result['generated_answer'],
                'iteration': 1,
                'all_queries': str([result['original_query']])
            })
        self.results_df = pd.DataFrame(df_data)
        return self.results_df

    def save_results(self, _ret="post_analysis", _k=None, _task='nq_test', _model='generator'):
        if _k is None:
            _k =  self.cutoff
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cutoff_label = "dynamic" if self.dynamic else f"cutoff_{self.cutoff}"
        json_filename = f"{self.output_dir}/analysis_results_{cutoff_label}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.result, f, indent=2, ensure_ascii=False)
        
        results_df = self.prepare_results()
        if self.ans_data is not None:
            logger.info("Using pre-built logging methods for evaluation...")
            self.log_fn(results_df, _ret, self.ans_data, _k, _task, _model)
            
            eval_results = self.evaluator(results_df, self.ans_data)
            
            logger.info("=== SUMMARY STATISTICS (using your evaluator) ===")
            logger.info(f"Total queries processed: {len(results_df)}")
            logger.info(f"Average EM Score: {eval_results['em'].mean():.3f}")
            logger.info(f"Average F1 Score: {eval_results['f1'].mean():.3f}")
            logger.info(f"EM Score > 0: {(eval_results['em'] > 0).sum()}/{len(eval_results)}")
            logger.info(f"F1 Score > 0: {(eval_results['f1'] > 0).sum()}/{len(eval_results)}")

        if self.dynamic:
            cutoff_stats = {}
            for qid, data in self.extracted_data.items():
                cutoff_used = data['cutoff_used']
                cutoff_stats[cutoff_used] = cutoff_stats.get(cutoff_used, 0) + 1

            logger.info("==== Dynamic Cutoff Statistics ====")
            for cutoff, count in sorted(cutoff_stats.items()):
                logger.info(f"Cutoff {cutoff}: {count} queries")
            
        else:
            logger.warning("Answer data not provided. Saving results without evaluation.")
            csv_filename = f"{self.output_dir}/analysis_summary_{cutoff_label}_{timestamp}.csv"
            results_df.to_csv(csv_filename, index=False)
            logger.info(f"Results saved to: {csv_filename}")
            
        logger.info(f"Detailed results saved to: {json_filename}")
        logger.info("Results have been logged using your existing infrastructure!")

    def evaluator(self, res, _ans):
        df_content = []
        
        for qid in res.qid.unique():
            golden_answers = _ans[_ans.qid == qid].gold_answer.to_list()
            if not golden_answers:
                logger.warning(f"No golden answers found for QID: {qid}")
                df_content.append([qid, 0.0, 0.0])
                continue
            golden_answers = [str(a) for a in golden_answers if pd.notna(a) and a is not None and str(a).strip() != '']
            
            if not golden_answers:
                logger.warning(f"All golden answers are NaN/None for QID: {qid}")
                df_content.append([qid, 0.0, 0.0])
                continue
            isnull_indicator = res[res.qid == qid].qanswer.isnull().values[0]
            if isnull_indicator:
                df_content.append([qid, 0.0, 0.0])
                continue
                
            prediction = res[res.qid == qid].qanswer.values[0]
            if prediction is None or prediction == "" or pd.isna(prediction):
                df_content.append([qid, 0.0, 0.0])
                continue
            
            prediction = str(prediction).strip()
            
            if prediction == "":
                df_content.append([qid, 0.0, 0.0])
                continue
            
            try:
                em_score = pyterrier_rag._measures.ems(prediction, golden_answers)
                
                f1_list = []
                for a in golden_answers:
                    f1_list.append(pyterrier_rag._measures.f1_score(prediction, a))
                f1_score = max(f1_list) if f1_list else 0.0
                
                df_content.append([qid, em_score, f1_score])
                
            except Exception as e:
                logger.error(f"Error evaluating QID {qid}: {str(e)}")
                logger.error(f"Prediction type: {type(prediction)}, value: '{prediction}'")
                logger.error(f"Golden answers types: {[type(a) for a in golden_answers]}, values: {golden_answers}")
                df_content.append([qid, 0.0, 0.0])
        
        return pd.DataFrame(df_content, columns=['qid', 'em', 'f1'])

    def log_fn(self, res, _ret, _ans, _k=3, _task='nq_test', _model='r1'):
        if(_task=='nq_test'):
            output_filename = f"{_ret}_{_k}_{_model}.res"
        else:
            output_filename = f"{_ret}_{_k}_{_task}_{_model}.res"
        csvfile = pathlib.Path(f"./res/{output_filename}")
        eval_csvfile = pathlib.Path(f"./eval_res/{output_filename}")
        res.to_csv(f"./res/{output_filename}", mode='a', index=False, header=not csvfile.exists())
        eval_res = self.evaluator(res, _ans)  # FIXED: Added self
        eval_res.to_csv(f"./eval_res/{output_filename}", mode='a', index=False, header=not eval_csvfile.exists())
        return res

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--task", type=str, default="nq_test")
    parser.add_argument("--retriever", type=str)
    args = parser.parse_args()

    cutoff = args.cutoff
    dynamic = args.dynamic
    task = args.task
    retriever = args.retriever

    if retriever == "bm25":
        res_filepath = "./datasets/res/bm25_3_hotpotqa_dev_r1.res"
        qpp_filepath = "./datasets/qpp_res/bm25_3_hotpotqa_dev_r1.res"
    else:
        res_filepath = "./datasets/res/E5_3_hotpotqa_dev_r1.res"
        qpp_filepath = "./datasets/qpp_res/E5_3_hotpotqa_dev_r1.res"
    
    analysis = PostAnalysis(
        res_filepath=res_filepath,
        qpp_filepath=qpp_filepath,
        cutoff=cutoff,
        dynamic=dynamic,
        task=task
    )

    analysis.generating_answers()

    model_name = f"dynamic_{cutoff}" if dynamic else f"fixed_{cutoff}"
    
    analysis.save_results(
        _ret="post_analysis", 
        _k=cutoff, 
        _task="nq_test" if task == "nq_test" else "hotpot_qa", 
        _model=model_name
    )
