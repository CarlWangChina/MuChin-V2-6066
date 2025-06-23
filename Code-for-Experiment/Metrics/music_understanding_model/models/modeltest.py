import torch

def calculate_accuracy_and_recall(model, dataset):
    model.eval()
    total_samples = len(dataset)
    correct_samples = 0
    true_positives = 0
    actual_positives = 0
    with torch.no_grad():
        for i in range(total_samples):
            inputs, labels = dataset[i]
            inputs = inputs.unsqueeze(0)
            outputs = model(inputs)
            predicted_labels = torch.argmax(outputs, dim=1)
            if predicted_labels == labels:
                correct_samples += 1
            if predicted_labels == 1 and labels == 1:
                true_positives += 1
            if labels == 1:
                actual_positives += 1
    accuracy = correct_samples / total_samples
    recall = true_positives / actual_positives if actual_positives >0 else 0
    return accuracy, recall