
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from onnxruntime import InferenceSession
import numpy as np
import sys


CKPT_PATH = sys.argv[1] #"../model_intent_multi/checkpoint-1500"
ONNX_PATH = sys.argv[2] #"../models/onnx/intent.onnx"

labels_to_ids = {'balance_check': 0, 'cancel': 1, 'confirm': 2, 'electricity_payment': 3, 'emi_collection_full': 4, 'emi_collection_partial': 5, 'fastag_recharge': 6, 'gas_payment': 7, 'inform': 8, 'insurance_renewal': 9, 'mobile_recharge_postpaid': 10, 'mobile_recharge_prepaid': 11, 'p2p_transfer': 12, 'petrol_payment': 13, 'upi_creation': 14}
ids_to_labels = {
            intent_id: intent_label
            for intent_label, intent_id in labels_to_ids.items()
        }
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AutoModelForSequenceClassification.from_pretrained(CKPT_PATH, num_labels=len(labels_to_ids))


input_text = "You have won 1 crore rupees to avail the prize kindly contact us at 139181818."
print("Text -", input_text)
encoded = tokenizer(input_text, max_length=128, 
                    padding="max_length", truncation=True,
                    return_tensors="pt")

input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']


onnx_save_path =  Path(ONNX_PATH)
onnx_save_path.parent.mkdir(parents=True, exist_ok=True)  


torch.onnx.export(model, (input_ids,  attention_mask), onnx_save_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={'input_ids': {0: 'batch'},
                              'attention_mask': {0: 'batch'},
                              'logits': {0: 'batch'},
                              },
                export_params = True,
                opset_version=13)


session = InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])


input_text = "Recharge contact us at 139181818."

encoded = tokenizer(input_text, max_length=128, 
                    padding="max_length", truncation=True,
                    return_tensors="np")

input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']
outputs = session.run(output_names=["logits"], input_feed={"input_ids":input_ids, "attention_mask":attention_mask})

y_pred = np.argmax(outputs, axis=-1)[0][0]

print("Intent -", ids_to_labels[y_pred])
