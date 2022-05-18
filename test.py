import transformers
import textattack
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
from textattack import Attacker
from textattack.attack_recipes import PWWSRen2019, CLARE2020, DeepWordBugGao2018, CheckList2020
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper

# Load pretrain model and tokenizer
model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
attack = DeepWordBugGao2018.build(model_wrapper)

# Tran/test
train_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="train")
eval_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")


ADVERSARIAL_TRAINING = False
if ADVERSARIAL_TRAINING:
    ##### Adversarial Training #####

    # Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
    training_args = textattack.TrainingArgs(
        num_epochs=3,
        num_clean_epochs=1,
        num_train_adv_examples=100,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        log_to_tb=True,
    )

    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
        attack,
        train_dataset,
        eval_dataset,
        training_args
    )
    trainer.train()

####### Attack ######
# Attack 20 samples with CSV logging and checkpoint saved every 5 interval
attack_args = textattack.AttackArgs(num_successful_examples=1, log_to_csv="log.csv", checkpoint_interval=5, checkpoint_dir="checkpoints", disable_stdout=True)
attacker = textattack.Attacker(attack, eval_dataset, attack_args)
AttackResults = attacker.attack_dataset()

# Results
original = AttackResults[0].diff_color(color_method='file')[0]
purturbed =AttackResults[0].diff_color(color_method='file')[1]
print('-----original------:')
print(original)
print('-----purturbed------:')
print(purturbed)