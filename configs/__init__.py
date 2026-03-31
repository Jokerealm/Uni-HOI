

def get_default_config():
    "load default config"
    from hydra import compose, initialize

    with initialize(version_base='1.1', config_path="./", job_name="sample"):
        cfg = compose(config_name="config",
                      # overrides=["db=mysql", "db.user=me"] # override configs wrote in the file
                      )
    return cfg
