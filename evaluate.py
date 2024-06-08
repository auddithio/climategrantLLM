import numpy as np
import torch
import faiss
from beir.retrieval.evaluation import EvaluateRetrieval

def top_k_metrics(filename, index, queries, label_indices, inst_uids, label_uids, topk_list=[1, 3, 5]):
    topk = 56
    # print('topk', topk)
    D, I = index.search(queries, topk)
    
    qrels = {}
    results = {}
    for idx, targets in enumerate(label_indices):
        # adding top k labels into the dictionary
        qrels[str(inst_uids[idx])] = {str(label_uids[t]): 1 for t in targets}

        # ### ADDED FOR DEBUGGING! 
        # for k in range(topk):
        #   print('k:', k)
        #   print({label_uids[I[idx][k]]: float(D[idx][k])})
        # print('inst_uids[idx]', inst_uids[idx])
        # print('results:', results)
        # print('results[inst_uids[idx]]', results[inst_uids[idx]])
        # ###
        # adding top k results into the dictionary
        results[str(inst_uids[idx])] = {str(label_uids[I[idx][k]]): float(D[idx][k]) for k in range(topk)}
        
        # ### ADDED FOR DEBUGGING! 
        # for k in range(topk):
        #     label_index = I[idx][k]
        #     if label_index in label_uids:
        #         results[inst_uids[idx]][label_uids[label_index]] = float(D[idx][k])
        #     else:
        #         print(f"Warning: label_index {label_index} not found in label_uids")
        #         print(f"inst_uids[idx]: {inst_uids[idx]}, idx: {idx}, k: {k}")
        # ####

    with open(f'accuracy_by_class_{filename}_eval_only.txt', 'w') as f:
        for title, dictionary in results.items():
            f.write(f"{title}: {dictionary}\n")
    
    # Define the range of indices
    max_index = 56

    # Step 2: Create a 2D numpy array with the appropriate shape
    num_rows = len(results)
    num_cols = max_index+1
    preds = np.zeros((num_rows, num_cols))

    # Step 3: Fill in the weights from the dictionaries
    row_index = 0
    for key in results.keys():
        sub_dict = results[key]
        for idx_str, value in sub_dict.items():
            idx = int(idx_str)
            preds[row_index, idx] = value
        row_index += 1

    return EvaluateRetrieval.evaluate(qrels, results, topk_list), preds


# CHANGED !

def accuracy_per_class(preds, target_label_indices, uid_title_map, filename, k=5):
    class_accuracies = {}
    
    # Iterate over each instance
    for i, indices in enumerate(target_label_indices):
        # Get the top k predicted labels for the current instance
        top_k_indices = np.argsort(-preds[i])[:k]
        
        # Check if any of the top k indices match the target label indices
        matched_indices = set(indices) & set(top_k_indices)
        
        # Increment accuracy count for the matched indices
        for index in indices:
            if index in class_accuracies:
                if index in matched_indices:
                    class_accuracies[index] += 1
            else:
                if index in matched_indices:
                    class_accuracies[index] = 1
                else:
                    class_accuracies[index] = 0
    
    # Normalize accuracy counts to get accuracy values
    for label in class_accuracies:
        total_instances = len([indices for indices in target_label_indices if label in indices])
        accuracy = class_accuracies[label] / total_instances
        class_accuracies[label] = accuracy
    
    # Get title corresponding to each class
    class_accuracies_with_title = {uid_title_map[label]: accuracy for label, accuracy in class_accuracies.items()}
    
    # Write accuracies to file
    with open(filename, 'w') as f:
        for title, accuracy in class_accuracies_with_title.items():
            f.write(f"{title}: {accuracy},\n")
    
    return class_accuracies_with_title

def evaluate(epoch, experiment, uid_title_map, tokenizer, model, titles, contents, labels, target_label_indices, inst_uids, label_uids, batch_size=1560, device=torch.device('cuda'), print_freq=100):
    torch.cuda.empty_cache()
    model.eval()
    
    sent = np.array([titles[i] + '\t' + contents[i] for i in range(len(titles))], dtype=object)

    idx = 0
    n = len(titles)
    inst_emb = []
    print('Embedding instances...')
    with torch.no_grad():
        while idx < n:
            tokens = tokenizer(sent[np.arange(idx, min(idx + batch_size, n))].tolist(), padding=True, truncation=True, return_tensors="pt")
            for k in tokens:
                tokens[k] = tokens[k].to(device)
            emb = model(**tokens).last_hidden_state[:, 0]
            inst_emb.append(emb.cpu())
            
            if (idx // batch_size) % print_freq == 0:
                print(f'{idx}/{n}')

            idx += batch_size

    idx = 0
    n = len(labels)
    labels = np.array(labels, dtype=object)
    label_emb = []
    print('Embedding labels...')
    with torch.no_grad():
        while idx < n:
            tokens = tokenizer(labels[np.arange(idx, min(idx + batch_size, n))].tolist(), padding=True, truncation=True, return_tensors="pt")
            for k in tokens:
                tokens[k] = tokens[k].to(device)
            emb = model(**tokens).last_hidden_state[:, 0]
            label_emb.append(emb.cpu())
            
            if (idx // batch_size) % print_freq == 0:
                print(f'{idx}/{n}')

            idx += batch_size

    inst_emb = torch.cat(inst_emb, dim=0).numpy()
    label_emb = torch.cat(label_emb, dim=0).numpy()

    faiss.normalize_L2(inst_emb)
    faiss.normalize_L2(label_emb)

    index = faiss.IndexFlatIP(inst_emb.shape[1])
    index.add(label_emb)
 
    top_k = [1, 3, 5]

    metrics, preds = top_k_metrics( 'babyeval', index, inst_emb, target_label_indices, inst_uids, label_uids, topk_list=top_k)
    
    print('metrics!!!', metrics)
    
    for m in metrics:
        print(m)

    
    # Compute accuracy per class
    print(preds)
    print(target_label_indices)
    # updated to include target_label_indices instead of true_labels
    class_accuracies = accuracy_per_class(preds, target_label_indices, uid_title_map, filename=f"accuracy_by_class/{experiment}_{epoch}.txt")
    return metrics[-1]['P@1']
