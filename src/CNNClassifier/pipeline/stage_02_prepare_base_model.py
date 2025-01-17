from src.CNNClassifier.config.configuration import ConfigurationManager
from src.CNNClassifier.components.prepare_base_model import PrepareBaseModel
from src.CNNClassifier import logger

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        final_model = prepare_base_model.prepare_full_model()

STAGE_NAME = "Prepare base model"

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} completed <<<<<\n\nx==================x")
    except Exception as e:
        logger.exception(e)
        raise e