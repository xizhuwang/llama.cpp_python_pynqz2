# llama.cpp_python_pynqz2
以下為「[abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)」以下為在 **ARM (cortex-a9)** 環境下安裝並使用 **llama-cpp-python** 的 **完整流程** 說明，包含如何在原始碼中註解特定檔案以避免動態函式庫出錯，以及在 32 位元系統上無法讀取超過 4GB 模型檔時的建議：

---

## 1. 安裝 Python 端所需套件

1. 建議先升級 `pip`：
   ```bash
   pip install --upgrade pip
   ```
2. 安裝必要套件：
   ```bash
   pip install llama-cpp-python tqdm numpy diskcache
   ```
3. 若需要從 Hugging Face 下載模型，安裝 `huggingface_hub`：
   ```bash
   pip install huggingface_hub
   ```
4. （可選）檢查套件安裝與版本：
   ```bash
   pip show llama-cpp-python
   pip show huggingface_hub
   pip show tqdm
   pip show numpy
   pip show diskcache
   ```

> **注意**  
> - 若無法以 root 身分安裝，可在以上指令後面加上 `--user`。  
> - 在 x86_64 架構上，通常直接安裝即可正常使用，不需自行編譯；但在 **ARM (cortex-a9)** 上若想最佳化，需要進行以下步驟。

---

## 2. 修改檔案（跳過部分檔案以避免編譯出錯）

由於部分 ARMv7 平台（cortex-a9）可能在編譯 `sgemm.cpp` 會報錯（牽涉到 ARMv8.2 FP16 指令），需先行移除或註解。
1. 進入目錄：
   ```bash
   cd ~/llama-cpp-python/vendor/llama.cpp/ggml/src/ggml-cpu
   vim CMakeLists.txt
   ```
2. 找到原先 `list(APPEND GGML_CPU_SOURCES ... sgemm.cpp sgemm.h)` 等處，將其註解：
   ```cmake
   # list(APPEND GGML_CPU_SOURCES
   #     ggml-cpu/llamafile/sgemm.cpp
   #     ggml-cpu/llamafile/sgemm.h
   # )
   ```
3. 編輯 `ggml-cpu.c` (若還有呼叫 sgemm 的區段)，同樣把呼叫位置註解掉或改成空函式。
```
vim ggml-cpu.c
 ```
 
### 2.2 重新編譯

1. 回到 `llama-cpp-python` 主目錄：
   ```bash
   cd ~/llama-cpp-python
   ```
2. 若之前已安裝舊版本，先卸載：
   ```bash
   pip uninstall llama-cpp-python
   ```
3. 編譯並安裝：針對 ARMv7 (cortex-a9 + hard float + NEON + OpenMP)：
   ```bash
   CMAKE_ARGS="-DCMAKE_C_FLAGS='-march=armv7-a -mfloat-abi=hard -mfpu=neon -O3 -fopenmp' \
              -DCMAKE_CXX_FLAGS='-march=armv7-a -mfloat-abi=hard -mfpu=neon -O3 -fopenmp'" \
   pip install .
   ```
   這會自動將編好的 `.so`（含 `libllama.so`、`libggml.so` 等）安裝到您的 Python site-packages 中。

---

## 3. 下載 Hugging Face 模型並測試

### 3.1 下載模型

1. 若尚未安裝，請執行：
   ```bash
   pip install huggingface_hub
   ```
2. 登入 Hugging Face：
   ```bash
   huggingface-cli login
   huggingface-cli whoami
   ```
   確認您已登入自己的 Hugging Face 帳號。
3. 下載模型：
   ```
   mkdir -p ~/models/tiny-llama-4bit
   cd ~/models/tiny-llama-4bit
   ```
   ```
   huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF \
   --local-dir . \
   --local-dir-use-symlinks False
下載完成後，請確認該資料夾內出現 `.bin`、`.ggml`、`.gguf` 等模型檔。

> 若您要測試的模型在：
> ```
> ~/models/llama3-taide-lx-8b-chat-alpha1-4bit/taide-8b-a.3-q4_k_m.gguf
> ```
> 請確認該檔已確實存在於此路徑。
---

### 3.2 撰寫測試程式 (test_model.py)

以下是一個最簡易的示範程式，請確定 `model_path` 指到您實際的模型檔名：

```python
import sys
import os
from llama_cpp import Llama

def main():
    # 請將下面這段路徑修改為實際想使用的 gguf 檔案
    model_path = os.path.expanduser("~/models/tiny-llama-4bit/tinyllama-1.1b-chat-v0.3.Q4_0.gguf")

    # 初始化模型
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,       # 上下文窗口大小
            temperature=0.7,  # 控制生成文本的多樣性
            top_p=0.9         # nucleus sampling 的參數
        )
        print("模型加載成功！")
    except FileNotFoundError as e:
        print(f"模型文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"初始化模型時發生錯誤: {e}")
        sys.exit(1)

    # 初始化對話上下文
    conversation = []

    print("連續對話開始（輸入 'exit' 結束對話）")
    while True:
        try:
            user_input = input("您: ")
            if user_input.lower() == "exit":
                print("結束對話")
                break

            # 將使用者輸入加入上下文
            conversation.append(f"您: {user_input}")

            # 組合完整對話作為提示
            prompt = "\n".join(conversation) + "\nAI:"

            # 生成模型回應
            output = llm(prompt, max_tokens=100)
            response = output["choices"][0]["text"].strip()

            # 將模型回應加入上下文
            conversation.append(f"AI: {response}")

            # 輸出模型回應
            print(f"AI: {response}")
        except Exception as e:
            print(f"發生錯誤: {e}")
            break

if __name__ == "__main__":
    main()
```

---

### 3.3 執行測試程式

```bash
python test_model.py
```
若載入成功，顯示類似：
```
模型加載成功！
連續對話開始（輸入 'exit' 結束對話）
您:
```
即可與模型互動。
---

## 4. 在 32 位元系統載入超過 4GB 檔案的限制

在 **ARMv7** (32-bit) 環境下，如果模型檔超過 2GB～4GB，執行時常會出現：
```
Value too large for defined data type
```
這代表系統檔案操作受到 32 位元檔案偏移量的限制，導致無法打開此大檔案。常見解法如下：

1. **換用更小的模型**  
   - 如果模型已壓縮到 `<2GB`，可避免大檔案讀取限制。
2. **使用 64 位元系統 / 硬體**  
   - 在 64-bit OS (e.g. aarch64, x86_64) 上載入 4GB+ 檔案不會受到同樣限制。
3. **嘗試 Large File Support**  
   - 在 32-bit Linux 下啟用 `-D_FILE_OFFSET_BITS=64`，並確保所有函式庫、程式碼都正確處理超過 2GB 的檔案。但實務中並不容易成功，而且若系統記憶體也不足，跑大模型仍會卡住。

因此，**若您的檔案約 4.6GB**，且必須在 32 位元 (ARMv7) 環境使用，建議改用更小的模型或換用 64 位元平台。

---

## 5. 總結

1. **安裝 `llama-cpp-python`**  
   - 先安裝 Python 套件與必要依賴  
   - 修改 `CMakeLists.txt`、`ggml-cpu.c` 等，以移除 `sgemm.cpp` 相關段落  
   - 編譯安裝指定編譯旗標（cortex-a9 / hard float / NEON / OpenMP）  

2. **下載並測試模型**  
   - 使用 Hugging Face CLI 下載相應的 `.gguf`、`.ggml` 或 `.bin` 檔  
   - 在 `test_model.py` 中設定正確 `model_path`，執行互動測試  

3. **若遇到「Value too large for defined data type」**  
   - 表示在 32-bit OS 上檔案超過 2GB 或 4GB，無法讀取  
   - **實用解法**：換更小模型，或換用 64 位元系統。  

如上流程，即可在 **ARM (cortex-a9)** 平台完成 **llama-cpp-python** 的安裝與使用；若您有大檔案模型（>4GB）需求，建議升級至 64 位元平台或使用更小模型。

