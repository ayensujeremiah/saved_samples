import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizer
from PIL import Image
from io import BytesIO
import boto3
import json
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from botocore.exceptions import ParamValidationError
from training_config import MAX_LENGTH, IMAGE_SIZE, PATCH_SIZE


LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

BUCKET_NAME = "mock-cogvlm-test"
IMAGE_PREFIX = "data/images/"
JSON_PREFIX = "data/json/"
IGNORE_INDEX = -100




def load_data_from_s3(buckets_with_prefixes):
    s3_client = boto3.client('s3')
    data = []
    for bucket_name, prefix in buckets_with_prefixes:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith('.json'):
                        try:
                            response = s3_client.get_object(Bucket=bucket_name, Key=key)
                            content = response['Body'].read().decode('utf-8')
                            data.extend(json.loads(content))
                        except ParamValidationError as e:
                            print(f"Parameter validation error for key {key}: {e}")
                        except Exception as e:
                            print(f"Error fetching object {key} from bucket {bucket_name}: {e}")
    return data



# def load_data_from_s3(buckets_with_prefixes):
#     s3_client = boto3.client('s3')
#     data = []
#     for bucket_name, prefix in buckets_with_prefixes:
#         paginator = s3_client.get_paginator('list_objects_v2')
#         page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

#         for page in page_iterator:
#             if "Contents" in page:
#                 for obj in page["Contents"]:
#                     key = obj["Key"]
#                     if key.endswith('.json'):
#                         response = s3_client.get_object(Bucket=bucket_name, Key=key)
#                         content = response['Body'].read().decode('utf-8')
#                         data.extend(json.loads(content))
#     return data


buckets_with_prefixes = [
    #("mock-cogvlm-test", "data/json/")
    ("cogagent-finetuning", "data/json/train/"),  # Med bucket and prefix
    ("cogagent-finetuning-non-med", "data/json/train/")  # Non-med bucket and prefix
]


image_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cross_transform = transforms.Compose([
    transforms.Resize((1120, 1120), interpolation=transforms.InterpolationMode.BICUBIC), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])







# def load_data_from_s3(bucket_name, prefix):
#     s3_client = boto3.client('s3')
#     data = []
#     paginator = s3_client.get_paginator('list_objects_v2')
#     page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
#     for page in page_iterator:
#         if "Contents" in page:
#             for obj in page["Contents"]:
#                 key = obj["Key"]
#                 if key.endswith('.json'):  
#                     response = s3_client.get_object(Bucket=bucket_name, Key=key)
#                     content = response['Body'].read().decode('utf-8')
#                     data.extend(json.loads(content))
#     return data



class MinoChatDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 prefix=JSON_PREFIX,
                 buckets_with_prefixes=buckets_with_prefixes,
                 #bucket_name=BUCKET_NAME, 
                 image_transform=image_transform, 
                 cross_transform=cross_transform,
                 image_size=IMAGE_SIZE, 
                 patch_size=PATCH_SIZE, 
                 max_length=MAX_LENGTH):
        
        self.data = load_data_from_s3(buckets_with_prefixes=buckets_with_prefixes)
        #self.data = load_data_from_s3(bucket_name=bucket_name, prefix=prefix)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.cross_transform = cross_transform
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_length = max_length
        self.s3_client = boto3.client('s3')
        

    def load_image_from_s3(self, image_s3_path):
        try:
            bucket, key = self.parse_s3_path(image_s3_path)
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            image_data = obj['Body'].read()
            image = Image.open(BytesIO(image_data)).convert('RGB')
            return image
        except Exception as e:
            print(f"Failed to load image from {image_s3_path}: {e}")
            return None

    
    def transform_image(self, path):
        raw_image = self.load_image_from_s3(path)
        if raw_image is not None:
            image = self.image_transform(raw_image)
            cross_image = self.cross_transform(raw_image)
            return image, cross_image
        return None, None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        conversation_key = 'conversation' if 'conversation' in item else 'conversations'
        conversation = item.get(conversation_key, [])
        #print(conversation)
        
        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        
        data_dict = {}
        
        
        if "image" in item:
            image_path = item['image']
            image, cross_image = self.transform_image(image_path)
            
        else:
            image = torch.zeros((3, 224, 224), dtype=torch.float16)
            cross_image = torch.zeros((3, 1120, 1120), dtype=torch.float16)
            
            
        if image is None and cross_image is None:
            image = torch.zeros((3, 224, 224), dtype=torch.float16)
            cross_image = torch.zeros((3, 1120, 1120), dtype=torch.float16)
            image = [image]
            cross_image = [cross_image]
            vision_token_num = ((self.image_size // self.patch_size) ** 2) + 2
            input_ids += [self.tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
            data_dict['images'] = image
            data_dict['cross_images'] = cross_image
            
        
        else:
            image = [image]
            cross_image = [cross_image]
            vision_token_num = ((self.image_size // self.patch_size) ** 2) + 2
            input_ids += [self.tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
            data_dict['images'] = image
            data_dict['cross_images'] = cross_image
            

            
        # Text processing with special tokens
        text_with_special_tokens = self.prepare_text_with_special_tokens(conversation)
        #print(text_with_special_tokens)

        # Tokenize text
        text_encoding = self.tokenizer(text_with_special_tokens, truncation=True, max_length=self.max_length, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze(0)
        
        #print(f"encodings: {text_encoding}")
        
        

        input_ids += text_encoding.tolist()
        #print(input_ids)
        
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_encoding)
        
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        

        # Assuming `input_ids` is your tensor of token IDs for a given sequence
        tokens = input_ids

        # Token IDs for "USER:" and "ASSISTANT:"
        user_token_id = torch.tensor([3148, 1001, 29901], dtype=torch.long)
        assistant_token_id = torch.tensor([319, 1799, 9047, 13566, 29901], dtype=torch.long)

        start_indices = []
        end_indices = []

        i = 0
        while i < len(tokens):
            # Check if current segment starts with "USER:"
            if torch.equal(tokens[i:i+len(user_token_id)], user_token_id):
                start_pos = i + len(user_token_id)
                i += len(user_token_id)  # Move past the "USER:" tokens

                # Look for the next "ASSISTANT:" to mark the end of masking
                while i < len(tokens) and not torch.equal(tokens[i:i+len(assistant_token_id)], assistant_token_id):
                    i += 1

                # Set end index right before "ASSISTANT:" starts
                end_indices.append(i)
                start_indices.append(start_pos)
            i += 1

        # Initialize a mask with False (not to be masked)
        mask = torch.full_like(tokens, False, dtype=torch.bool)

        # Set the mask to True for each segment identified
        for start, end in zip(start_indices, end_indices):
            mask[start:end] = True

        # Apply mask, setting tokens to IGNORE_INDEX where mask is True
        labels = torch.where(mask, torch.tensor(IGNORE_INDEX, dtype=torch.long), tokens)
        #print(f"Labels: {labels}")
       
        attention_mask = [1] * len(input_ids)

        data_dict['input_ids'] = input_ids
        data_dict['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        data_dict['labels'] = labels
        data_dict['token_type_ids'] = token_type_ids


        return data_dict
    
    
    @staticmethod
    def parse_s3_path(s3_path):
        """Extract bucket and key from an S3 path."""
        if not s3_path.startswith("s3://"):
            raise ValueError("Invalid S3 path format. Must start with s3://")
        _, _, bucket_key = s3_path.partition("s3://")
        bucket, _, key = bucket_key.partition("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 path: {s3_path}")
        return bucket, key

    
    
    def prepare_text_with_special_tokens(self, conversation):
        text_with_special_tokens = ""
        if not isinstance(conversation, list):
            print("Conversation is not a list.")
            return text_with_special_tokens

        for i in range(0, len(conversation), 2):  # Process in steps of 2 to handle question-answer pairs
            try:
                human_exchange = conversation[i]
                if i + 1 < len(conversation):
                    gpt_exchange = conversation[i + 1]
                else:
                    print(f"No GPT exchange for human exchange at index {i}. Skipping.")
                    continue

                # Check if both exchanges are dictionaries with a 'value' key
                if not all(isinstance(exchange, dict) and 'value' in exchange for exchange in [human_exchange, gpt_exchange]):
                    print(f"Malformed exchange at indices {i} or {i+1}. Skipping.")
                    continue

                # Extract and clean text, excluding "<image>" tokens
                human_text = str(human_exchange['value']).replace("<image>", "").strip()
                gpt_text = str(gpt_exchange['value']).replace("<image>", "").strip()

                # Concatenate the cleaned text with "user" and "assistant" labels
                text_with_special_tokens += f"<EOI> USER: {human_text} ASSISTANT: {gpt_text} </s> "
            except Exception as e:
                print(f"Error processing conversation at indices {i} or {i+1}: {e}. Skipping.")

        text_with_special_tokens = text_with_special_tokens.strip()
        return text_with_special_tokens

