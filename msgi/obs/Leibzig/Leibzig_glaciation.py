from glaciation_time_estimator.auxiliary_func.config_reader import read_config
from  glaciation_time_estimator.data_postprocessing.Extract_glaciations import extract_glaciation_events, gen_glac_df
import pandas as pd
import os
if __name__=="__main__":
    
    config = read_config()
    global global_rmse
    global_rmse = config["Global_sqrt_mse"]
    parquet_dir = os.path.join(
            config["postprocessing_output_dir"],
            "Leibzig_data",
            config["time_folder_name"],
        )

    combined_cloud_df = pd.read_parquet(f"{parquet_dir}.parquet")
    result_df = extract_glaciation_events(combined_cloud_df, glac_tresh=config.get("glac_threshold",0.4))
    if result_df.empty:
        raise ValueError("No glaciation events found for the year")
    print("Generating glaciations dataframe")
    glaciations_df = gen_glac_df(result_df, combined_cloud_df)
    glac_df_path = f"{parquet_dir}_glac.parquet"
    print(f"Writing glaciations to parquet: {glac_df_path}")
    glaciations_df.to_parquet(glac_df_path)