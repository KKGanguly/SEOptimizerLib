import pandas as pd
import numpy as np
from pathlib import Path

def analyze_winners(directory="scott_knott"):
    files = list(Path(directory).glob("*.csv"))
    
    budget_groups = {
        '<= 50': [30, 50],
        '100': [100],
        '200': [200]
    }
    
    summary_data = []
    
    for f in sorted(files):
        df = pd.read_csv(f)
        tier_cols = [c for c in df.columns if c.endswith('_tier')]
        if not tier_cols:
            continue
            
        row = {"Matchup File": f.stem.replace('scott_knott_dataset_', '')}
        
        for bg_name, b_list in budget_groups.items():
            opt_stats = {} # Will hold {opt_name: (mean, std)}
            
            optimizers_in_file = list(set([c.split('-')[0] for c in tier_cols]))
            
            for opt in optimizers_in_file:
                tiers_for_opt = []
                for b in b_list:
                    target_col = f"{opt}-{b}_tier"
                    if target_col in df.columns:
                        vals = pd.to_numeric(df[target_col], errors='coerce').dropna()
                        tiers_for_opt.extend(vals.tolist())
                
                if tiers_for_opt:
                    avg_tier = np.mean(tiers_for_opt)
                    # Calculate std dev (0 if only one data point)
                    std_tier = np.std(tiers_for_opt, ddof=1) if len(tiers_for_opt) > 1 else 0.0
                    opt_stats[opt] = (avg_tier, std_tier)
            
            if not opt_stats:
                row[bg_name] = "Bypassed/N/A"
            else:
                # Sort first by mean (x[1][0]), then by std dev (x[1][1])
                sorted_opts = sorted(opt_stats.items(), key=lambda x: (x[1][0], x[1][1]))
                
                best_opt = sorted_opts[0][0]
                best_mean = sorted_opts[0][1][0]
                best_std = sorted_opts[0][1][1]
                
                # Check for strict ties on the mean
                ties = [opt for opt, stats in sorted_opts if stats[0] == best_mean]
                
                if len(ties) > 1:
                    # If there's a tie on the mean, the sort already put the lowest std dev first.
                    # We can highlight that it was a tie broken by variance.
                    row[bg_name] = f"{best_opt} ({best_mean:.2f} ± {best_std:.2f}) *"
                else:
                    row[bg_name] = f"{best_opt} ({best_mean:.2f} ± {best_std:.2f})"
                    
        summary_data.append(row)
        
    summary_df = pd.DataFrame(summary_data)
    print("\n--- TOURNAMENT WINNERS (Mean ± Std) ---")
    print("Note: * indicates a tie on the mean that was broken by lower standard deviation.\n")
    print(summary_df.to_markdown(index=False))
    
    output_file = "tournament_stage_winners.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved clean results to {output_file}")

if __name__ == "__main__":
    analyze_winners()