## 專案結構

- `quickstart/experiment.ipynb`: 主要的 Jupyter Notebook 文件，包含訓練、驗證和測試
- `my.yaml`: 模型和訓練設定配置文件
- `eval.py`: 由助教提供的評估腳本
-  驗證結果將保存在 `valid_results.json` 文件中
-  測試結果將保存在 `test_results.json` 文件中

## 使用說明

1. **訓練**
   - 使用預訓練的 RT-DETR-l 模型
   - 在自定義數據集上微調模型（使用 `my.yaml` 配置）

2. **驗證**
   - 在驗證集上運行模型
   - 處理結果並保存為 JSON 格式
   - 使用 `eval.py` 腳本評估結果

3. **測試**
   - 在測試集上運行模型
   - 處理結果並保存為 JSON 格式

## 運行步驟

1. 打開 `quickstart/experiment.ipynb` 文件
2. 按順序運行所有單元格
3. 訓練完成後，模型將自動在驗證集和測試集上運行
4. 驗證結果將保存在 `valid_results.json` 文件中
5. 測試結果將保存在 `test_results.json` 文件中