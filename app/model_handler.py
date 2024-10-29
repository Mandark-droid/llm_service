from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from torch.utils.data import DataLoader
import re

class LLMService:
    def __init__(self, model_name="gpt2"):
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Set the padding token to be the same as the EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def fine_tune(self, data_pairs):
        # Check if data_pairs is provided and contains data
        if not data_pairs:
            raise ValueError("Data pairs are empty. Please check the data source.")

        print("Fine-tuning with the following data pairs:")
        # Validate the structure of data_pairs
        for item in data_pairs:
            if not isinstance(item, dict) or "prompt" not in item or "response" not in item:
                raise ValueError("Data pair items must be dictionaries with 'prompt' and 'response' keys.")

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict({
            "prompt": [item["prompt"] for item in data_pairs],
            "response": [item["response"] for item in data_pairs]
        })

        # Define the tokenize function to concatenate prompt and response
        def tokenize_function(examples):
            combined_text = [f"{p} {r}" for p, r in zip(examples["prompt"], examples["response"])]
            return self.tokenizer(combined_text, padding="max_length", truncation=True, max_length=256)

        # Tokenize the dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "response"])

        # Create input_ids and labels for loss calculation
        def create_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples

        # Create labels
        tokenized_dataset = tokenized_dataset.map(create_labels, batched=True)

        # Check tokenized dataset size
        if len(tokenized_dataset) == 0:
            raise ValueError("Tokenized dataset is empty. Please check input data or tokenization.")

        # Create a DataLoader for the tokenized dataset
        train_dataloader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            num_train_epochs=1,
            logging_dir='./logs',
            remove_unused_columns=False
        )

        # Initialize the Trainer with the tokenized dataset
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset
        )

        # Train the model
        trainer.train()

    def generate_code(self, prompt, max_length=100, temperature=0.7, top_k=50):
        # Ensure the prompt is properly formatted
        formatted_prompt = f"{prompt.strip()}\n\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        # Generate output
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,  # Get a single sequence
            do_sample=True  # Enable sampling
        )

        # Decode the output and clean it
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove potential repetitions or unwanted patterns
        return self.clean_output(generated_code)

    def clean_output(self, generated_code):
        # Remove unwanted phrases or repetitions
        clean_code = ' '.join(list(dict.fromkeys(generated_code.split())))  # Remove duplicates
        # Simple regex to retain only Python-like code snippets
        code_lines = clean_code.splitlines()
        code_lines = [line for line in code_lines if
                      re.match(r'^\s*def\s+\w+\(.*\):', line) or line.strip().startswith("import")]
        return "\n".join(code_lines).strip()


# Example usage
if __name__ == "__main__":
    # Sample data pairs for testing
    sample_data_pairs = [
        {"prompt": "Write a Python function to add two numbers.", "response": "def add(a, b): return a + b"},
        {"prompt": "What is the capital of France?", "response": "The capital of France is Paris."}
    ]

    print("Sample Data Pairs:", sample_data_pairs)  # Debugging line

    # Create an instance of the LLMService and fine-tune with sample data
    llm_service = LLMService()

    try:
        llm_service.fine_tune(sample_data_pairs)
        print("Fine-tuning completed successfully.")

        # Test code generation
        generated_code = llm_service.generate_code("Write a function to multiply two numbers.")
        print("Generated Code:", generated_code)
    except ValueError as e:
        print(f"Error during fine-tuning: {e}")
