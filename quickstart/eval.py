import json
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def convert_to_tensor(data):
    if len(data["boxes"]) == 0:
        return {"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)}
    
    boxes = torch.tensor(data["boxes"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.int64)
    return {"boxes": boxes, "labels": labels}

def convert_predictions_to_tensor(data):
    if len(data["boxes"]) == 0:
        return {"boxes": torch.empty((0, 4), dtype=torch.float32), "scores": torch.empty((0,), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)}
    
    boxes = torch.tensor(data["boxes"], dtype=torch.float32)
    scores = torch.tensor(data.get("scores", [1.0] * len(data["boxes"])), dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.int64)
    return {"boxes": boxes, "scores": scores, "labels": labels}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <predic_json> <target_json>")
        sys.exit(1)

    predic_json_file = sys.argv[1]
    target_json_file = sys.argv[2]

    predictions_json = load_json(predic_json_file)
    targets_json = load_json(target_json_file)

    # 我加的 避免json檔案順序布一樣會影響評估結果
    ''' 
    prediction_keys = set(predictions_json.keys())
    target_keys = set(targets_json.keys())

    # 檢查兩個檔案中的鍵是否完全相同
    if prediction_keys != target_keys:
        print("警告：預測檔案和目標檔案中的圖像不完全匹配。")
        print("只會評估兩個檔案中都存在的圖像。")

    # 使用兩個檔案共有的鍵，並進行排序
    common_keys = sorted(prediction_keys.intersection(target_keys))

    predictions = [convert_predictions_to_tensor(predictions_json[key]) for key in common_keys]
    targets = [convert_to_tensor(targets_json[key]) for key in common_keys]
    '''
    
    predictions = [convert_predictions_to_tensor(pred) for pred in predictions_json.values()]
    targets = [convert_to_tensor(target) for target in targets_json.values()]

    metric = MeanAveragePrecision()

    metric.update(predictions, targets)

    result = metric.compute()

    print("Evaluation Results:")
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1: 
                value = value.item() 
            else:
                value = value.tolist() 
        print(f"{key}: {value}")

    