import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
# Paper plot style (central definition: eval_pipeline/plot_style.py).
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "eval_pipeline")))
try:
    from plot_style import apply_paper_style as _apply_paper_style
    _apply_paper_style()
except Exception:
    pass


# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
AXIS_CONFIG = {
    "Background": {
        "L": {"xlim": (50, 225), "ylim": (0.001, 1), "log": True, "bw": 3},
        "A": {"xlim": (127, 129), "ylim": (0.001, 100), "log": True, "bw": 3},
        "B": {"xlim": (127, 129), "ylim": (0.001, 100), "log": True, "bw": 3}
    },
    "Foreground": {
        "L": {"xlim": (50, 225), "ylim": (None, None), "log": False, "bw": 1},
        "A": {"xlim": (100, 175), "ylim": (None, None), "log": False, "bw": 1},
        "B": {"xlim": (100, 175), "ylim": (None, None), "log": False, "bw": 1}
    },
    "Global": {
        "L": {"xlim": (0, 255), "ylim": (None, None), "log": True, "bw": 2},
        "A": {"xlim": (0, 255), "ylim": (None, None), "log": True, "bw": 2},
        "B": {"xlim": (0, 255), "ylim": (None, None), "log": True, "bw": 2}
    }
}

CHANNEL_LABELS = {
    "L": "Lightness",
    "A": "<-- Green | Red -->",
    "B": "<-- Blue | Yellow -->"
}

class ColorComparator:
    def load_datasets(self, real_path, model_data):
        all_dfs = []
        
        # Load Real GT
        if os.path.exists(real_path):
            df_real = pd.read_csv(real_path)
            df_real["Type"] = "Real (GT)"
            all_dfs.append(df_real)
        else:
            logger.error(f"Real CSV not found: {real_path}")

        # Load Models
        for path, name in model_data:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["Type"] = name
                all_dfs.append(df)
            else:
                logger.warning(f"Model CSV not found: {path}")

        if not all_dfs:
            return pd.DataFrame()
        
        # Concat fills missing columns with NaN
        return pd.concat(all_dfs, ignore_index=True)

    def plot_comparison(self, df, layer_name, prefix, output_filename):
        logger.info(f"Generating plot for {layer_name} -> {output_filename}")
        
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
        
        layer_conf = AXIS_CONFIG.get(layer_name, AXIS_CONFIG.get("Global", {}))
        channels = ["L", "A", "B"]
        
        plots_drawn = 0
        handles, labels = [], []

        for i, chan in enumerate(channels):
            ax = axes[i]
            col_name = f"{prefix}_{chan}"
            
            # Check if column exists at all
            if col_name not in df.columns:
                ax.text(0.5, 0.5, f"Missing Data\n({col_name})", ha='center', va='center')
                continue

            # Filter valid data for this column
            df_subset = df.dropna(subset=[col_name])
            
            if df_subset.empty:
                ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center')
                continue
            
            # Check which types remain after dropping NaNs
            remaining_types = df_subset["Type"].unique()
            if len(remaining_types) < len(df["Type"].unique()):
                missing = set(df["Type"].unique()) - set(remaining_types)
                logger.warning(f"  [Warning] {layer_name} ({chan}): Dropped {missing} due to missing data.")

            # Get config
            conf = layer_conf.get(chan, {"xlim": (None, None), "ylim": (None, None), "log": False, "bw": 1})

            try:
                sns.kdeplot(
                    data=df_subset, x=col_name, hue="Type", 
                    fill=True, alpha=0.1, linewidth=2, ax=ax, 
                    common_norm=False, warn_singular=False,
                    bw_adjust=conf.get("bw", 1) 
                )
                
                # Extract legend for the unified one, then remove local
                if ax.get_legend() is not None:
                    if not handles:
                        handles = ax.get_legend().legendHandles if hasattr(ax.get_legend(), 'legendHandles') else ax.get_legend().legend_handles
                        labels = [t.get_text() for t in ax.get_legend().texts]
                    ax.get_legend().remove()
                    
                plots_drawn += 1
            except Exception as e:
                logger.warning(f"KDE Plot failed for {col_name}: {e}")

            ax.set_title(f"{chan}-Channel Distribution")
            ax.set_xlabel(CHANNEL_LABELS.get(chan, f"Pixel Value ({chan})"))
            
            # Axis
            if conf.get("xlim") and conf["xlim"] != (None, None): ax.set_xlim(conf["xlim"])
            if conf.get("ylim") and conf["ylim"] != (None, None): ax.set_ylim(conf["ylim"])
            if conf.get("log"): 
                try: ax.set_yscale('log')
                except: pass
            
            if chan in ["A", "B"]:
                ax.axvline(128, color='gray', linestyle=':', alpha=0.5)

        if plots_drawn > 0:
            if handles and labels:
                fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
            plt.suptitle(f"{layer_name} Color Distribution", fontsize=16)
            plt.tight_layout()
            plt.savefig(output_filename.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
            logger.info(f"Saved {output_filename}")
        else:
            logger.warning(f"No valid data for {layer_name}, skipping save.")
        plt.close()

    def plot_combined_comparison(self, df, layers, output_filename):
        layer_names = [l[0] for l in layers]
        logger.info(f"Generating combined plot for {layer_names} -> {output_filename}")
        
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(len(layers), 3, figsize=(20, 6 * len(layers)), sharey=False)
        
        channels = ["L", "A", "B"]
        plots_drawn = 0
        handles, labels = [], []

        for row_idx, (layer_name, prefix) in enumerate(layers):
            layer_conf = AXIS_CONFIG.get(layer_name, AXIS_CONFIG.get("Global", {}))
            
            for col_idx, chan in enumerate(channels):
                ax = axes[row_idx, col_idx]
                col_name = f"{prefix}_{chan}"
                
                if col_name not in df.columns:
                    ax.text(0.5, 0.5, f"Missing Data\n({col_name})", ha='center', va='center')
                    continue

                df_subset = df.dropna(subset=[col_name])
                
                if df_subset.empty:
                    ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center')
                    continue
                
                conf = layer_conf.get(chan, {"xlim": (None, None), "ylim": (None, None), "log": False, "bw": 1})

                try:
                    sns.kdeplot(
                        data=df_subset, x=col_name, hue="Type", 
                        fill=True, alpha=0.1, linewidth=2, ax=ax, 
                        common_norm=False, warn_singular=False,
                        bw_adjust=conf.get("bw", 1) 
                    )
                    
                    if ax.get_legend() is not None:
                        if not handles:
                            handles = ax.get_legend().legendHandles if hasattr(ax.get_legend(), 'legendHandles') else ax.get_legend().legend_handles
                            labels = [t.get_text() for t in ax.get_legend().texts]
                        ax.get_legend().remove()
                        
                    plots_drawn += 1
                except Exception as e:
                    logger.warning(f"KDE Plot failed for {col_name}: {e}")

                ax.set_title(f"{layer_name} - {chan}-Channel")
                ax.set_xlabel(CHANNEL_LABELS.get(chan, f"Pixel Value ({chan})"))
                
                if conf.get("xlim") and conf["xlim"] != (None, None): ax.set_xlim(conf["xlim"])
                if conf.get("ylim") and conf["ylim"] != (None, None): ax.set_ylim(conf["ylim"])
                if conf.get("log"): 
                    try: ax.set_yscale('log')
                    except: pass
                
                if chan in ["A", "B"]:
                    ax.axvline(128, color='gray', linestyle=':', alpha=0.5)

        if plots_drawn > 0:
            if handles and labels:
                fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
            
            plt.tight_layout()
            plt.savefig(output_filename.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
            logger.info(f"Saved {output_filename}")
        else:
            logger.warning(f"No valid data, skipping save.")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_csv", required=True)
    parser.add_argument("--model_csvs", nargs='+', required=True)
    parser.add_argument("--model_names", nargs='+', required=True)
    parser.add_argument("--out_dir", default="comparison_plots")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load Data
    csvs = [c for c in args.model_csvs if c != '+']
    names = [n for n in args.model_names if n != '+']
    
    if len(csvs) != len(names):
        logger.error(f"Mismatch: {len(csvs)} CSVs vs {len(names)} names.")
        exit(1)

    model_data = list(zip(csvs, names))
    comparator = ColorComparator()
    df_all = comparator.load_datasets(args.real_csv, model_data)
    
    if not df_all.empty:
        # 1. Combined Background and Foreground
        combined_layers = [("Background", "BG"), ("Foreground", "FG")]
        comparator.plot_combined_comparison(df_all, combined_layers, os.path.join(args.out_dir, "compare_bg_fg_combined.png"))

        # 2. Global (Vergleich ALLE, auch Unlabeled)
        comparator.plot_comparison(df_all, "Global", "Global", os.path.join(args.out_dir, "compare_global.png"))
    else:
        logger.error("Dataset empty.")