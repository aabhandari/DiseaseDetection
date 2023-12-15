
from data_preparation import prepare_data
from model_creation import create_models
from train_evaluate import train_and_evaluate
from ensemble_learning import ensemble_predict

def main():
    image_paths, labels = prepare_data()
    models = create_models()
    trained_models = train_and_evaluate(models, image_paths, labels)
    test_data = [...]  # Define test data
    predictions = ensemble_predict(trained_models, test_data)

if __name__ == "__main__":
    main()
