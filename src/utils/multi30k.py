import sys
sys.path.append('..')
import torch
from tqdm import tqdm
import pickle as pkl
import os
from data_utils import *
from collate_fns import *
from dataset import DocDataset
from torch.utils.data import DataLoader
from ddp import *

def get_Multi30k(params, model, test = ('2017', 'mscoco'), force_pretraining = False):
	print(f"[DEBUG]: force_pretraining: {force_pretraining}")
	print(f"[DEBUG]: params.src_lang: {params.src_lang}, params.tgt_lang: {params.tgt_lang}")
	print(f"[DEBUG]: test set: {test}")

	if force_pretraining:
		langs = ['en'] # Only pretraining on en images
		test = ('2017', 'mscoco') # Anyway not going to be used
	else:
		langs =  [params.src_lang, params.tgt_lang]

	print(f"[DEBUG]: Languages to process: {langs}")


	datapath = os.path.join(params.data_dir, 'multi30k')

	print(f"[DEBUG]: Data path: {datapath}")


	os.makedirs(os.path.join(datapath, f'text/data/task1/mbart'), exist_ok = True)
	os.makedirs(os.path.join(datapath, f'text/data/task1/{params.image_encoder}'), exist_ok = True)
	# Reading train files
	train_texts = {lang: open(os.path.join(datapath, f'text/data/task1/raw/train.{lang}')).read().splitlines() for lang in langs}
	
	print(f"[DEBUG]: Loaded train_texts for languages: {list(train_texts.keys())}")

	try:
		train_tok_mbart = {lang: pkl.load(open(os.path.join(datapath, f'text/data/task1/mbart/train.{lang}.pkl'), 'rb')) for lang in langs}
		print("[DEBUG]: Loaded train tokenized data for mbart.")
	except Exception as e:
		print(f"[DEBUG]: Could not load mbart train tokenized data: {e}")
		train_tok_mbart = {lang: tokenize(train_texts[lang], model.tokenizer, lang, os.path.join(datapath, f'text/data/task1/mbart/train.{lang}.pkl'), f'Tokenizing train {lang} with mbart') for lang in langs}
		print("[DEBUG]: Created mbart tokenized train data.")

	try:
		train_tok_mclip = {lang: pkl.load(open(os.path.join(datapath, f'text/data/task1/{params.image_encoder}/train.{lang}.pkl'), 'rb')) for lang in langs}
		print("[DEBUG]: Loaded train tokenized data for mclip.")
	except Exception as e:
		print(f"[DEBUG]: Could not load mclip train tokenized data: {e}")
		train_tok_mclip = {lang: tokenize(train_texts[lang], model.clip.text_preprocessor, lang, os.path.join(datapath, f'text/data/task1/{params.image_encoder}/train.{lang}.pkl'), f'Tokenizing train {lang} with {params.image_encoder}') for lang in langs}
		print("[DEBUG]: Created mclip tokenized train data.")
	
	# Reading test files
	test_texts = {lang: open(os.path.join(datapath, f'text/data/task1/raw/test_{test[0]}_{test[1]}.{lang}')).read().splitlines() for lang in langs}
	print(f"[DEBUG]: Loaded test_texts for languages: {list(test_texts.keys())}")
	try:
		test_tok_mbart = {lang: pkl.load(open(os.path.join(datapath, f'text/data/task1/mbart/test_{test[0]}_{test[1]}.{lang}.pkl'), 'rb')) for lang in langs}
		print("[DEBUG]: Loaded test tokenized data for mbart.")
	except Exception as e:
		print(f"Could not load mbart test tokenized data: {e}")
		test_tok_mbart = {lang: tokenize(test_texts[lang], model.tokenizer, lang, os.path.join(datapath, f'text/data/task1/mbart/test_{test[0]}_{test[1]}.{lang}.pkl'), f'Tokenizing test {lang} with mbart') for lang in langs}
		print("[DEBUG]: Created mbart tokenized test data.")

	try:
		test_tok_mclip = {lang: pkl.load(open(os.path.join(datapath, f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pkl'), 'rb')) for lang in langs}
		print("[DEBUG]: Loaded test tokenized data for mclip.")
	except Exception as e:
		print(f"[DEBUG]: Could not load mclip test tokenized data: {e}")
		test_tok_mclip = {lang: tokenize(test_texts[lang], model.clip.text_preprocessor, lang, os.path.join(datapath, f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pkl'), f'Tokenizing test {lang} with {params.image_encoder}') for lang in langs}
		print("[DEBUG]: Created mclip tokenized test data.")

	train_image_splits = open(os.path.join(datapath, f'text/data/task1/image_splits/train.txt')).read().splitlines()
	test_image_splits = open(os.path.join(datapath, f'text/data/task1/image_splits/test_{test[0]}_{test[1]}.txt')).read().splitlines()

	# Getting images and embedding with CLIP. Same for text
	print('[DEBUG]: Loaded all text files. Getting images...')
	train_img_embs = get_image_embs(model.clip, os.path.join(datapath, 'images/train'), train_image_splits, os.path.join(datapath, f'text/data/task1/{params.image_encoder}/train.pth'), 'Embedding train images', model.clip.image_preprocessor)
	test_img_embs = get_image_embs(model.clip, os.path.join(datapath, f'images/test_{test[0]}_{test[1]}'), test_image_splits, os.path.join(datapath, f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.pth'), f'Embedding test_{test[0]}_{test[1]} images', model.clip.image_preprocessor)
	
	train_text_embs, test_text_embs = {}, {}
	for lang in langs:
		print(f"[DEBUG]: Processing embeddings for language: {lang}")
		embs_f = os.path.join(datapath, f'text/data/task1/{params.image_encoder}/train.{lang}.pth')
		print(f"[DEBUG]: Looking for train embeddings file: {embs_f}")
		try:
			train_text_embs[lang] = torch.load(embs_f)
			print(f"[DEBUG]: Loaded train embeddings for {lang} from {embs_f}")
		except Exception as e:
			print(f"[DEBUG]: Could not load train embeddings for {lang}: {e}")
			print(f"[DEBUG]: Creating embeddings for train.{lang}...")
			text_ds = DocDataset(train_tok_mclip[lang])
			print(f"[DEBUG]: Created dataset for train.{lang}, size: {len(text_ds)}")
			text_dl = DataLoader(text_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_texts)
			print(f"[DEBUG]: DataLoader for train.{lang} created with batch size 256")
			train_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding train.{lang} mclip')
			print(f"[DEBUG]: Saved train embeddings for {lang} to {embs_f}")

		# Test embeddings
		embs_f = os.path.join(datapath, f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pth')
		# print(f"[DEBUG]: Could not load test embeddings for {lang}: {e}")
		print(f"[DEBUG]: Creating embeddings for test_{test[0]}_{test[1]}.{lang}...")
		text_ds = DocDataset(test_tok_mclip[lang])
		print(f"[DEBUG]: Created dataset for test_{test[0]}_{test[1]}.{lang}, size: {len(text_ds)}")
		text_dl = DataLoader(text_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_texts)
		print(f"[DEBUG]: DataLoader for test_{test[0]}_{test[1]}.{lang} created with batch size 256")
		test_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding test_{test[0]}_{test[1]}.{lang} mclip')
		print(f"[DEBUG]: Saved test embeddings for {lang} to {embs_f}")
		# print(f"[DEBUG]: Looking for test embeddings file: {embs_f}")
		# try:
		# 	test_text_embs[lang] = torch.load(embs_f)
		# 	print(f"[DEBUG]: Loaded test embeddings for {lang} from {embs_f}")
		# except Exception as e:
		# 	print(f"[DEBUG]: Could not load test embeddings for {lang}: {e}")
		# 	print(f"[DEBUG]: Creating embeddings for test_{test[0]}_{test[1]}.{lang}...")
		# 	text_ds = DocDataset(test_tok_mclip[lang])
		# 	print(f"[DEBUG]: Created dataset for test_{test[0]}_{test[1]}.{lang}, size: {len(text_ds)}")
		# 	text_dl = DataLoader(text_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_texts)
		# 	print(f"[DEBUG]: DataLoader for test_{test[0]}_{test[1]}.{lang} created with batch size 256")
		# 	test_text_embs[lang] = create_embeddings(text_dl, model.clip, embs_f, f'Embedding test_{test[0]}_{test[1]}.{lang} mclip')
		# 	print(f"[DEBUG]: Saved test embeddings for {lang} to {embs_f}")

	return train_texts, test_texts, train_tok_mbart, test_tok_mbart, train_img_embs, test_img_embs, train_text_embs, test_text_embs, train_tok_mclip, test_tok_mclip
