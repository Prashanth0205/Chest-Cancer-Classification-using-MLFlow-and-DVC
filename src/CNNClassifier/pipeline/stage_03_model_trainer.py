from src.CNNClassifier.config.configuration import ConfigurationManager
from src.CNNClassifier.components.model_trainer import Training
from src.CNNClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_genertor()
        training.train()
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} completed <<<<<\n\nx==================x")
    except Exception as e:
        logger.exception(e)
        raise e