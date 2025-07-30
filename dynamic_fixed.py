"""
okay so to get this straight and summarize it in a nutshell.
1. I will run the code on full dataset to get the logs -- DONE
2. i will make a new script that is supposed to do the post run analysis.
3. Since we have the intermediate queries in the res folder and the qpp scores in the qpp_res folder, we can use these to do the analysis
4. For the cut off, we will choose for example 2, and then extract the intermediate queries, and feed them to a generator which is the qwen instruction 7b model with a prompt that i develop. The prompt will include the intermediate queries with a slight tweak like generate a short answer.
5. I will log the answers and then evaluate them and log everything.

"""

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
    def __init__(self, res_filepath, qpp_filepath, cutoff, dynamic=False, ans_filepath=None, output_dir='rag/analysis_results'):
        self.client = OpenAI(base_url="http://api.llm.apps.os.dcs.gla.ac.uk/v1", api_key=os.environ["IDA_LLM_API_KEY"])
        self.result = []
        self.output_dir = output_dir
        self.results_df = None

        # for dynamic cutoff
        self.qpp_filepath = qpp_filepath
        self.dynamic = dynamic
        self.qpp_df = None
        self.dynamic_cutoff = {}
        self.res_filepath = res_filepath
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
            # Get answers directly from PyTerrier dataset (default: NQ test)
            if not pt.java.started():
                pt.java.init()
            self.ans_data = pt.get_dataset('rag:nq').get_answers('test')
            logger.info("Loaded NQ test answers from PyTerrier dataset")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("./res", exist_ok=True)
        os.makedirs("./eval_res", exist_ok=True)

    def calculate_dynamic_cutoff(self):
        """
        Remember: (Qi - Qi-1)/ Qi-1 > 0.25
        """
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
        search_pattern = r"(<search>.*?</search>\s*<information>.*?</information>)"

        for _,res in res_df.iterrows():
            qid = res['qid']
            query = res['query']
            output = res['output']
            golden_answer = res['qanswer']

            if "<|im_start|>assistant" in output:
                output = output.split("<|im_start|>assistant", 1)[1]
            matches = re.findall(search_pattern, output, flags=re.DOTALL)

            if self.dynamic:
                current_cutoff = self.dynamic_cutoff[qid]
                if current_cutoff == res['iteration']:
                    print(f"for {qid}, dynamic cutoff is same as iteration")
            else:
                current_cutoff = self.cutoff
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
            search_pattern = r"<search>(.*?)</search>"
            search_part = re.search(search_pattern, query)
            search_term = search_part.group(1).strip() if search_part else "N/A"

            info_pattern = r"<information>(.*?)</information>"
            info_part = re.search(info_pattern, query, flags=re.DOTALL)
            if not info_part:
                print(query)
            info_term = info_part.group(1).strip() if info_part else "N/A"

            intermediate_content += f"<search> {search_term} </search>  <information> {info_term} </information>"
        prompt1 = f"""You are a helpful assistant. Based on the search results provided below, give a direct and concise answer to the question.

Search results:
{intermediate_content}

Instructions:
- Use only the information provided in the search results above
- If multiple results contain relevant information, synthesize them
- Give only the most accurate and specific answer possible
- Do not explain your reasoning or add extra information
- Generate the exact answer.
- Keep the answer short (3-4 words only)

Question: {original_query}
Answer:"""

        prompt2 = f"""
You are a helpful assistant. Answer the given question. You must conduct reasoning inside <think> and </think> first every time with the information. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Do not provide the reasoning or explain the thought process. Keep your answer short and as accurate as you can. For example, <answer> Beijing </answer>. Question: who got the first nobel prize in physics?

Question: {original_query}
Context: {intermediate_content}
"""
        return prompt2
        
    def generator_model(self, prompt, qid):

        try:
            result = self.client.completions.create(model='qwen-2.5-7b-instruct',prompt=prompt)
            raw_answer = result.choices[0].text.strip() if result.choices else "No Response"
            
            # Clean the generated answer
            generated_answer = raw_answer
            # Remove common prefixes and XML tags
            generated_answer = re.sub(r'^(The answer is|Answer:)\s*', '', generated_answer, flags=re.IGNORECASE)
            generated_answer = re.sub(r'</?answer>', '', generated_answer)
            generated_answer = generated_answer.strip()

            logger.info(f"Successfully generated answer for QID: {qid}")
            return {
                "generated_answer": generated_answer,
                "raw_answer": raw_answer,  # Keep raw for debugging
                "status":"success",
                "error":None
            }

        except Exception as e:
            logger.info(f"The answer was not generated successfully for QID {qid}: {str(e)}")
            return {
                "generated_answer":None,
                "status":"failed",
                "error":str(e)  # FIXED: Convert exception to string for JSON serialization
            }

    def generating_answers(self):
        logger.info(f"Starting analysis with cutoff: {'dynamic' if dynamic else self.cutoff} ")
        logger.info(f"Processing {len(self.extracted_data)} queries")
        for idx, (qid, data) in enumerate(self.extracted_data.items()):
            logger.info(f"Processing QID: {qid}")
            prompt = self.build_prompt(data['query'], data['inter_queries'])
            model_result = self.generator_model(prompt, qid)

            result_entry = {
                    "qid": qid,
                    "original_query": data['query'],
                    "golden_answer": data['golden_answer'],
                    "num_intermediate_queries": len(data['inter_queries']),
                    "intermediate_queries": data['inter_queries'],
                    "prompt": prompt,
                    "generated_answer": model_result['generated_answer'],
                    "model_status": model_result['status'],
                    "model_error": model_result['error']
                }
            self.result.append(result_entry)
            print(f"QID: {qid}")
            # print(f"Prompt:{prompt}")
            print(f"Original Query: {data['query']}")
            print(f"Golden Answer: '{data['golden_answer']}'")
            print(f"Generated Answer: '{model_result['generated_answer']}'")
            print(f"Generated Answer Length: {len(model_result['generated_answer']) if model_result['generated_answer'] else 0}")
            print(f"Exact Match: {data['golden_answer'] == model_result['generated_answer']}")
            print(f"Intermediate Queries Count: {len(data['inter_queries'])}")
            print("-" * 80)

    def prepare_results(self):  # FIXED: Method name matches the call
        df_data = []
        for result in self.result:
            df_data.append({
                'qid': result['qid'],
                'query': result['original_query'],
                'qanswer': result['generated_answer'],
                'output': result['generated_answer'],  # For compatibility
                'iteration': 1,
                'all_queries': str([result['original_query']])
            })
        self.results_df = pd.DataFrame(df_data)
        return self.results_df

    def save_results(self, _ret="post_analysis", _k=None, _task='nq_test', _model='generator'):
        """
        Save results using both new format and your existing logging methods
        """
        if _k is None:
            _k =  self.cutoff
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to JSON (new format)
        cutoff_label = "dynamic" if self.dynamic else f"cutoff_{self.cutoff}"
        json_filename = f"{self.output_dir}/analysis_results_{cutoff_label}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.result, f, indent=2, ensure_ascii=False)
        
        # Prepare DataFrame in your standard format
        results_df = self.prepare_results()  # FIXED: Method name corrected
        
        # Use your existing logging methods if answer data is available
        if self.ans_data is not None:
            # Use your pre-built log_fn method
            logger.info("Using pre-built logging methods for evaluation...")
            self.log_fn(results_df, _ret, self.ans_data, _k, _task, _model)
            
            # Calculate and display summary using your evaluator
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
            # Save summary CSV manually
            csv_filename = f"{self.output_dir}/analysis_summary_{cutoff_label}_{timestamp}.csv"
            results_df.to_csv(csv_filename, index=False)
            logger.info(f"Results saved to: {csv_filename}")
            
        logger.info(f"Detailed results saved to: {json_filename}")
        logger.info("Results have been logged using your existing infrastructure!")

    def evaluator(self, res, _ans):
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
    
    args = parser.parse_args()

    cutoff = args.cutoff
    dynamic = args.dynamic
    
    res_filepath = "./res/bm25_3_r1.res"
    qpp_filepath = "./qpp_res/bm25_3_r1.res"
    
    analysis = PostAnalysis(
        res_filepath=res_filepath,
        qpp_filepath=qpp_filepath,
        cutoff=cutoff,
        dynamic=dynamic
    )
    
    analysis.generating_answers()

    model_name = f"dynamic_{cutoff}" if dynamic else f"fixed_{cutoff}"
    
    analysis.save_results(
        _ret="post_analysis", 
        _k=cutoff, 
        _task="nq_test", 
        _model=model_name
    )