import os
import torch
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, AutoModelForCausalLM
from torchvision import transforms
from data_preprocess3 import MinoChatDataset
import deepspeed
from deepspeed import zero
import json
from yarn_test import Yarn
import torch.nn.functional as F
import os
import shutil
import boto3
import io
from training_config import TOKENIZER_NAME, MODEL_NAME, BATCH_SIZE, NUM_EPOCHS, S3_MODEL_DIR, SAVE_MODEL_NAME, MAX_LENGTH, CHECKPOINT_DIR


os.environ['DS_SKIP_CUDA_CHECK'] = "1"




from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, tokenizer):
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels =  [item['labels'] for item in batch]
    images = [item['images'] for item in batch if 'images' in item and item['images'] is not None]

    cross_images = [item['cross_images'] for item in batch if 'cross_images' in item and item['cross_images'] is not None]
  
    # Pad sequences as before
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    token_type_ids_padded = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    
    
    print(f"Labels shape: {labels_padded.shape}")
    print(f"Input Ids shape: {input_ids_padded.shape}")
    print(f"Attention Mask shape: {attention_mask_padded.shape}")
    print(f"Token shape: {token_type_ids_padded.shape}")


    # Construct the batched data dictionary
    batched_data = {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'token_type_ids': token_type_ids_padded,
    }

    if images:
        batched_data['images'] = images
    if cross_images:
        batched_data['cross_images'] = cross_images
   

    return batched_data



def train():
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get('LOCAL_RANK', 0))
    
    
    
    torch.cuda.set_device(rank)
    deepspeed.init_distributed()

    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir = "/home/ec2-user/SageMaker/model_cache/")
    
    dataset = MinoChatDataset(tokenizer=tokenizer)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    data_loader = DataLoader(dataset, 
                             batch_size=BATCH_SIZE, 
                             sampler=sampler, 
                             collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer))

    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                                 trust_remote_code=True,
                                                 cache_dir = "/home/ec2-user/SageMaker/model_cache/"
                                                 #load_in_4bit=True, 
                                                 #device_map="auto"
                                                )
    
#     for name, param in model.named_parameters():
#         param.requires_grad = False
    
    new_num_positions = 257
    model.config.vision_config['num_positions'] = new_num_positions

    #Directly modify the position_embedding in the PatchEmbedding layer
    vision_model = model.model.vision
    for layer in vision_model.children():
        if layer.__class__.__name__ == 'PatchEmbedding':
            new_position_embedding = torch.nn.Embedding(new_num_positions, layer.position_embedding.embedding_dim)
            new_position_embedding.weight.data[:257] = layer.position_embedding.weight.data[:257]
            layer.position_embedding = new_position_embedding

    
    
    model_engine, optimizer, _, _ = deepspeed.initialize(config_params="deepspeed_config3.json", model=model)

    for epoch in range(NUM_EPOCHS):
        model_engine.train()
        total_loss = 0
        for i, batch in enumerate(data_loader):
            #print( batch['input_ids'])
            
            input_ids = batch['input_ids'].to(rank)
            #print(f"input_ids: {input_ids}")
            attention_mask = batch['attention_mask'].to(rank)
            #print(f"attention_mask: {attention_mask}")
            labels = batch['labels'].to(rank)
            #print(f"labels: {labels}")
            token_type_ids=batch['token_type_ids'].to(rank)
            
            images_processed = []
            cross_images_processed = []
            
            
            #placeholder_img = torch.zeros((3, 224, 224), dtype=torch.bfloat16).to(rank)

            if 'images' in batch:  # Check if 'images' key exists in the batch
                for img_list in batch['images']:
                    processed_imgs = []  
                    if img_list is not None:
                        processed_imgs = [img.to(rank).bfloat16() for img in img_list]
                        images_processed.append(processed_imgs)
                        
            if 'cross_images' in batch:  
                for cross_img_list in batch['cross_images']:
                    cross_processed_imgs = []  
                    if cross_img_list is not None:
                        cross_processed_imgs = [cross_img.to(rank).bfloat16() for cross_img in cross_img_list]
                        cross_images_processed.append(cross_processed_imgs)
                
#             for img_list in batch['images']:
#                 processed_imgs = [img.to(rank).bfloat16() for img in img_list]
#                 images_processed.append(processed_imgs)
         

                
#             for cross_img_list in batch['cross_images']:
#                 processed_cross_imgs = [cross_img.to(rank).half() for cross_img in cross_img_list]
#                 cross_images_processed.append(processed_cross_imgs)
                
                
            optimizer.zero_grad()
            
            model_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
            
            if images_processed:
                model_args["images"] = images_processed
                model_args["token_type_ids"] = token_type_ids
            
            if cross_images_processed:
                model_args["cross_images"] = cross_images_processed
                
                
            outputs = model_engine(**model_args)

#             outputs = model_engine(input_ids=input_ids, 
#                                    attention_mask=attention_mask, 
#                                    labels=labels, 
#                                    images=images_processed, 
#                                    #cross_images=cross_images_processed,
#                                    token_type_ids=token_type_ids)

            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Rank {rank}, Epoch {epoch + 1}, Step {i}, Loss: {loss.item()}")
                
#             if i % 1000 == 0:
#                 model_engine.save_checkpoint(save_dir=CHECKPOINT_DIR)

        avg_loss = total_loss / len(data_loader)
        print(f"Rank {rank}, Epoch {epoch + 1} completed. Average Loss: {avg_loss}")
        
    model_engine.save_16bit_model(save_dir=CHECKPOINT_DIR, save_filename=f"{SAVE_MODEL_NAME}.bin")

    return model_engine

        
        

if __name__ == "__main__":
    model_engine = train()
