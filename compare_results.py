import pandas as pd
import numpy as np
import json
import pathlib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
from collections import defaultdict

class ResultsComparator:
    def __init__(self, base_path="."):
        self.base_path = pathlib.Path(base_path)
        self.results = {}  # from res/
        self.evaluations = {}  # from eval_res/
        self.qpp_results = {}  # from qpp_res/
        self.retrieval_results = {}  # from retrieval_res/
        
    def load_all_results(self):
        """Load all results from the 4 specified folders"""
        print("Loading results from res, eval_res, qpp_res, and retrieval_res folders...")
        
        # Load main results
        self._load_results()
        
        # Load evaluation results
        self._load_evaluations()
        
        # Load QPP results
        self._load_qpp_results()
        
        # Load retrieval results
        self._load_retrieval_results()
        
        print(f"Loaded {len(self.results)} result files")
        print(f"Loaded {len(self.evaluations)} evaluation files")
        print(f"Loaded {len(self.qpp_results)} QPP result files")
        print(f"Loaded {len(self.retrieval_results)} retrieval result files")
    
    def _load_results(self):
        """Load results from res/ folder"""
        res_path = self.base_path / "res"
        
        if res_path.exists():
            for file_path in res_path.glob("*.res"):
                key = file_path.stem
                try:
                    df = pd.read_csv(file_path)
                    self.results[key] = df
                    print(f"Loaded result: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        else:
            exit("Couldn't find folder")
    
    def _load_evaluations(self):
        """Load evaluation results from eval_res/ folder"""
        eval_res_path = self.base_path / "eval_res"
        
        if eval_res_path.exists():
            for file_path in eval_res_path.glob("*.res"):
                key = file_path.stem
                try:
                    df = pd.read_csv(file_path)
                    self.evaluations[key] = df
                    print(f"Loaded evaluation: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        else:
            print("eval_res/ folder not found")
    
    def _load_qpp_results(self):
        """Load QPP results from qpp_res/ folder"""
        qpp_res_path = self.base_path / "qpp_res"
        
        if qpp_res_path.exists():
            for file_path in qpp_res_path.glob("*.res"):
                key = file_path.stem
                try:
                    df = pd.read_csv(file_path)
                    self.qpp_results[key] = df
                    print(f"Loaded QPP result: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        else:
            print("qpp_res/ folder not found")
    
    def _load_retrieval_results(self):
        """Load retrieval results from retrieval_res/ folder"""
        retrieval_res_path = self.base_path / "retrieval_res"
        
        if retrieval_res_path.exists():
            for file_path in retrieval_res_path.glob("*.res"):
                key = file_path.stem
                try:
                    df = pd.read_csv(file_path)
                    self.retrieval_results[key] = df
                    print(f"Loaded retrieval result: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        else:
            print("retrieval_res/ folder not found")
    
    def compare_evaluations(self):
        """Compare evaluation metrics across different configurations"""
        print("\n" + "="*50)
        print("EVALUATION METRICS COMPARISON")
        print("="*50)
        
        comparison_data = []
        
        for key, eval_df in self.evaluations.items():
            if 'em' in eval_df.columns and 'f1' in eval_df.columns:
                mean_em = eval_df['em'].mean()
                mean_f1 = eval_df['f1'].mean()
                std_em = eval_df['em'].std()
                std_f1 = eval_df['f1'].std()
                
                # Parse configuration details from filename
                config_parts = key.split('_')
                retriever = config_parts[0] if len(config_parts) > 0 else 'unknown'
                k_value = config_parts[1] if len(config_parts) > 1 else 'unknown'
                model = config_parts[-1] if len(config_parts) > 2 else 'unknown'
                task = '_'.join(config_parts[2:-1]) if len(config_parts) > 3 else 'nq_test'
                
                comparison_data.append({
                    'experiment': key,
                    'retriever': retriever,
                    'k': k_value,
                    'model': model,
                    'task': task,
                    'mean_em': mean_em,
                    'std_em': std_em,
                    'mean_f1': mean_f1,
                    'std_f1': std_f1,
                    'num_queries': len(eval_df)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('mean_em', ascending=False)
            
            print("\nRanked by Exact Match (EM) Score:")
            print(comparison_df[['experiment', 'retriever', 'k', 'model', 'mean_em', 'mean_f1', 'num_queries']].to_string(index=False, float_format='%.4f'))
            
            # Find best performing configuration
            best_em = comparison_df.iloc[0]
            best_f1 = comparison_df.loc[comparison_df['mean_f1'].idxmax()]
            
            print(f"\nBest EM Score: {best_em['experiment']} (EM: {best_em['mean_em']:.4f})")
            print(f"Best F1 Score: {best_f1['experiment']} (F1: {best_f1['mean_f1']:.4f})")
            
            return comparison_df
        else:
            print("No evaluation data found for comparison")
            return None
    
    def compare_retrievers(self):
        """Compare different retriever performance"""
        print("\n" + "="*50)
        print("RETRIEVER COMPARISON")
        print("="*50)
        
        # Group results by retriever type
        retriever_groups = defaultdict(list)
        
        for key, eval_df in self.evaluations.items():
            if 'em' in eval_df.columns and 'f1' in eval_df.columns:
                retriever = key.split('_')[0]
                retriever_groups[retriever].append({
                    'experiment': key,
                    'mean_em': eval_df['em'].mean(),
                    'mean_f1': eval_df['f1'].mean()
                })
        
        for retriever, experiments in retriever_groups.items():
            print(f"\n{retriever.upper()} Retriever:")
            for exp in experiments:
                print(f"  {exp['experiment']}: EM={exp['mean_em']:.4f}, F1={exp['mean_f1']:.4f}")
    
    def compare_models(self):
        """Compare different model performance (r1 vs r1s)"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        # Group results by model type
        model_groups = defaultdict(list)
        
        for key, eval_df in self.evaluations.items():
            if 'em' in eval_df.columns and 'f1' in eval_df.columns:
                parts = key.split('_')
                model = parts[-1] if len(parts) > 0 else 'unknown'
                model_groups[model].append({
                    'experiment': key,
                    'mean_em': eval_df['em'].mean(),
                    'mean_f1': eval_df['f1'].mean()
                })
        
        for model, experiments in model_groups.items():
            print(f"\n{model.upper()} Model:")
            for exp in experiments:
                print(f"  {exp['experiment']}: EM={exp['mean_em']:.4f}, F1={exp['mean_f1']:.4f}")
    
    def analyze_qpp_methods(self):
        """Analyze Query Performance Prediction methods"""
        print("\n" + "="*50)
        print("QPP METHODS ANALYSIS")
        print("="*50)
        
        for key, qpp_df in self.qpp_results.items():
            print(f"\n{key}:")
            print(f"  Total QPP entries: {len(qpp_df)}")
            
            if 'qpp_method' in qpp_df.columns:
                method_counts = qpp_df['qpp_method'].value_counts()
                print("  QPP Methods:")
                for method, count in method_counts.items():
                    print(f"    {method}: {count} queries")
                    
                    # Show statistics for each method
                    method_data = qpp_df[qpp_df['qpp_method'] == method]
                    if 'qpp_estimation' in method_data.columns:
                        estimations = pd.to_numeric(method_data['qpp_estimation'], errors='coerce')
                        print(f"      Mean estimation: {estimations.mean():.4f}")
                        print(f"      Std estimation: {estimations.std():.4f}")
    
    def analyze_retrieval_patterns(self):
        """Analyze retrieval result patterns"""
        print("\n" + "="*50)
        print("RETRIEVAL PATTERNS ANALYSIS")
        print("="*50)
        
        for key, retrieval_df in self.retrieval_results.items():
            print(f"\n{key}:")
            print(f"  Total retrieved documents: {len(retrieval_df)}")
            
            if 'score' in retrieval_df.columns:
                print(f"  Mean retrieval score: {retrieval_df['score'].mean():.4f}")
                print(f"  Score range: {retrieval_df['score'].min():.4f} - {retrieval_df['score'].max():.4f}")
            
            if 'qid' in retrieval_df.columns:
                unique_queries = retrieval_df['qid'].nunique()
                avg_docs_per_query = len(retrieval_df) / unique_queries
                print(f"  Unique queries: {unique_queries}")
                print(f"  Average docs per query: {avg_docs_per_query:.1f}")
    
    def create_visualizations(self, save_path="./comparison_plots"):
        """Create visualizations comparing results"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        pathlib.Path(save_path).mkdir(exist_ok=True)
        
        # Prepare data for plotting
        plot_data = []
        for key, eval_df in self.evaluations.items():
            if 'em' in eval_df.columns and 'f1' in eval_df.columns:
                config_parts = key.split('_')
                retriever = config_parts[0] if len(config_parts) > 0 else 'unknown'
                k_value = config_parts[1] if len(config_parts) > 1 else 'unknown'
                model = config_parts[-1] if len(config_parts) > 2 else 'unknown'
                
                for _, row in eval_df.iterrows():
                    plot_data.append({
                        'experiment': key,
                        'retriever': retriever,
                        'k': k_value,
                        'model': model,
                        'em': row['em'],
                        'f1': row['f1'],
                        'qid': row.get('qid', 'unknown')
                    })
        
        if not plot_data:
            print("No data available for visualization")
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # EM Score Distribution by Experiment
        sns.boxplot(data=plot_df, x='experiment', y='em', ax=axes[0,0])
        axes[0,0].set_title('EM Score Distribution by Experiment')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # F1 Score Distribution by Experiment
        sns.boxplot(data=plot_df, x='experiment', y='f1', ax=axes[0,1])
        axes[0,1].set_title('F1 Score Distribution by Experiment')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Retriever Comparison
        if len(plot_df['retriever'].unique()) > 1:
            sns.boxplot(data=plot_df, x='retriever', y='em', ax=axes[0,2])
            axes[0,2].set_title('EM Score by Retriever')
        else:
            axes[0,2].text(0.5, 0.5, 'Single Retriever', ha='center', va='center', transform=axes[0,2].transAxes)
        
        # Model Comparison
        if len(plot_df['model'].unique()) > 1:
            sns.boxplot(data=plot_df, x='model', y='em', ax=axes[1,0])
            axes[1,0].set_title('EM Score by Model')
        else:
            axes[1,0].text(0.5, 0.5, 'Single Model', ha='center', va='center', transform=axes[1,0].transAxes)
        
        # K Value Comparison
        if len(plot_df['k'].unique()) > 1:
            sns.boxplot(data=plot_df, x='k', y='em', ax=axes[1,1])
            axes[1,1].set_title('EM Score by K Value')
        else:
            axes[1,1].text(0.5, 0.5, 'Single K Value', ha='center', va='center', transform=axes[1,1].transAxes)
        
        # EM vs F1 Scatter
        for exp in plot_df['experiment'].unique():
            exp_data = plot_df[plot_df['experiment'] == exp]
            axes[1,2].scatter(exp_data['em'], exp_data['f1'], label=exp, alpha=0.6)
        
        axes[1,2].set_xlabel('EM Score')
        axes[1,2].set_ylabel('F1 Score')
        axes[1,2].set_title('EM vs F1 Score Correlation')
        axes[1,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/comparison_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create QPP analysis plot if data exists
        if self.qpp_results:
            self._create_qpp_plots(save_path)
        
        print(f"Plots saved to {save_path}/")
    
    def _create_qpp_plots(self, save_path):
        """Create QPP-specific plots"""
        qpp_plot_data = []
        
        for key, qpp_df in self.qpp_results.items():
            if 'qpp_method' in qpp_df.columns and 'qpp_estimation' in qpp_df.columns:
                for _, row in qpp_df.iterrows():
                    qpp_plot_data.append({
                        'experiment': key,
                        'method': row['qpp_method'],
                        'estimation': pd.to_numeric(row['qpp_estimation'], errors='coerce'),
                        'qid': row.get('qid', 'unknown')
                    })
        
        if qpp_plot_data:
            qpp_plot_df = pd.DataFrame(qpp_plot_data)
            qpp_plot_df = qpp_plot_df.dropna(subset=['estimation'])
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # QPP Method Comparison
            sns.boxplot(data=qpp_plot_df, x='method', y='estimation', ax=axes[0])
            axes[0].set_title('QPP Estimation Distribution by Method')
            axes[0].tick_params(axis='x', rotation=45)
            
            # QPP by Experiment
            sns.boxplot(data=qpp_plot_df, x='experiment', y='estimation', ax=axes[1])
            axes[1].set_title('QPP Estimation by Experiment')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/qpp_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, output_file="comparison_report.txt"):
        """Generate a comprehensive comparison report"""
        print("\n" + "="*50)
        print("GENERATING COMPARISON REPORT")
        print("="*50)
        
        with open(output_file, 'w') as f:
            f.write("EXPERIMENT RESULTS COMPARISON REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("DATA SUMMARY\n")
            f.write("-" * 15 + "\n")
            f.write(f"Results files: {len(self.results)}\n")
            f.write(f"Evaluation files: {len(self.evaluations)}\n")
            f.write(f"QPP files: {len(self.qpp_results)}\n")
            f.write(f"Retrieval files: {len(self.retrieval_results)}\n\n")
            
            # Evaluation metrics summary
            f.write("EVALUATION METRICS SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for key, eval_df in self.evaluations.items():
                if 'em' in eval_df.columns and 'f1' in eval_df.columns:
                    f.write(f"\n{key}:\n")
                    f.write(f"  Queries: {len(eval_df)}\n")
                    f.write(f"  Mean EM: {eval_df['em'].mean():.4f} (±{eval_df['em'].std():.4f})\n")
                    f.write(f"  Mean F1: {eval_df['f1'].mean():.4f} (±{eval_df['f1'].std():.4f})\n")
                    f.write(f"  EM > 0: {(eval_df['em'] > 0).sum()} ({(eval_df['em'] > 0).sum()/len(eval_df)*100:.1f}%)\n")
                    f.write(f"  F1 > 0: {(eval_df['f1'] > 0).sum()} ({(eval_df['f1'] > 0).sum()/len(eval_df)*100:.1f}%)\n")
            
            # QPP analysis
            f.write("\n\nQPP METHODS ANALYSIS\n")
            f.write("-" * 25 + "\n")
            
            for key, qpp_df in self.qpp_results.items():
                f.write(f"\n{key}:\n")
                f.write(f"  Total QPP entries: {len(qpp_df)}\n")
                
                if 'qpp_method' in qpp_df.columns:
                    method_counts = qpp_df['qpp_method'].value_counts()
                    f.write("  Methods:\n")
                    for method, count in method_counts.items():
                        f.write(f"    {method}: {count} queries\n")
            
            # Retrieval patterns
            f.write("\n\nRETRIEVAL PATTERNS\n")
            f.write("-" * 20 + "\n")
            
            for key, retrieval_df in self.retrieval_results.items():
                f.write(f"\n{key}:\n")
                f.write(f"  Total documents: {len(retrieval_df)}\n")
                
                if 'qid' in retrieval_df.columns:
                    unique_queries = retrieval_df['qid'].nunique()
                    avg_docs_per_query = len(retrieval_df) / unique_queries
                    f.write(f"  Unique queries: {unique_queries}\n")
                    f.write(f"  Avg docs per query: {avg_docs_per_query:.1f}\n")
                
                if 'score' in retrieval_df.columns:
                    f.write(f"  Mean score: {retrieval_df['score'].mean():.4f}\n")
        
        print(f"Report saved to {output_file}")
    
    def statistical_significance_test(self):
        """Perform statistical significance tests between configurations"""
        print("\n" + "="*50)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*50)
        
        try:
            from scipy import stats
            
            eval_keys = list(self.evaluations.keys())
            
            if len(eval_keys) >= 2:
                print("\nPairwise t-tests for EM scores:")
                for i, config1 in enumerate(eval_keys):
                    for config2 in eval_keys[i+1:]:
                        em1 = self.evaluations[config1]['em']
                        em2 = self.evaluations[config2]['em']
                        
                        # Ensure same length for paired t-test
                        min_len = min(len(em1), len(em2))
                        em1_trimmed = em1[:min_len]
                        em2_trimmed = em2[:min_len]
                        
                        t_stat, p_value = stats.ttest_rel(em1_trimmed, em2_trimmed)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        
                        print(f"{config1} vs {config2}: t={t_stat:.3f}, p={p_value:.4f} {significance}")
            
        except ImportError:
            print("scipy not available for statistical tests")
        except Exception as e:
            print(f"Error in statistical tests: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare experiment results from res, eval_res, qpp_res, and retrieval_res folders")
    parser.add_argument("--base_path", type=str, default=".", help="Base path where result folders are located")
    parser.add_argument("--output_report", type=str, default="comparison_report.txt", help="Output report filename")
    parser.add_argument("--plot_path", type=str, default="./comparison_plots", help="Path to save plots")
    parser.add_argument("--no_plots", action="store_true", help="Skip creating plots")
    
    args = parser.parse_args()
    
    # Create comparator instance
    comparator = ResultsComparator(args.base_path)
    
    # Load all results
    comparator.load_all_results()
    
    # Perform comparisons
    comparator.compare_evaluations()
    comparator.compare_retrievers()
    comparator.compare_models()
    comparator.analyze_qpp_methods()
    comparator.analyze_retrieval_patterns()
    comparator.statistical_significance_test()
    
    # Generate visualizations
    if not args.no_plots:
        comparator.create_visualizations(args.plot_path)
    
    # Generate report
    comparator.generate_report(args.output_report)
    
    print(f"\nComparison complete! Check {args.output_report} for detailed results.")

if __name__ == "__main__":
    main()