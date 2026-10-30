"""Entry point: runs the full training pipeline end to end. This is what a
GitHub Actions CD job (or a scheduled retrain) would invoke."""
from networksecurity.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    artifact = pipeline.run_pipeline()
    print(f"Training complete. Model saved to: {artifact.trained_model_file_path}")
    print(f"Test F1: {artifact.test_metric_artifact.f1_score:.4f}")
