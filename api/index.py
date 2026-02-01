# app.py
# 运行本地：python app.py
# 部署推荐：Render.com / Vercel / Railway / PythonAnywhere（免费层足够）

from flask import Flask, render_template_string, request, jsonify, Response, send_file
import os
from openai import OpenAI
import time
import pandas as pd
from io import BytesIO
import markdown

app = Flask(__name__)

# =====================================
# DeepSeek API 配置（必须通过环境变量设置，安全！）
# =====================================
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY（你的 DeepSeek 密钥）")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# =====================================
# 生成学习路径函数（流式）
# =====================================
def generate_learning_path(profession_name):
    if not profession_name.strip():
        yield "请输入有效的职业名称。"
        return

    prompt = f"""
你是一位经验丰富的职业发展规划专家和教育设计师。
请为职业 '{profession_name}' 设计一个完整、可执行的学习路径，结构清晰，包括以下内容：

1. **自学模块**（推荐书籍、在线课程、文档、视频等，按阶段排列）
2. **面授/线下课程**（如果适用，推荐知名机构、认证课程）
3. **练习任务**（每个阶段的实战项目、小练习、Kaggle/开源贡献等）
4. **辅导环节**（如何找到导师、加入社区、Code Review、Pair Programming等）
5. **教授他人活动**（写博客、做分享、带新人、创建教程等，用于巩固和输出）

同时生成：
- **技能标准**（初级/中级/高级分别需要掌握什么）
- **知识要素细目表**（核心知识点清单，可用表格形式）
- **评估规划**：从 知识掌握、技能应用、行为表现、业务结果 四个维度评估
- **师资规划**：专家、培训师、评估师、导师 的角色和获取方式

输出格式尽量使用 Markdown，结构清晰，便于阅读。
语言专业、鼓励性强，路径现实可行，时间估算合理（假设每周投入15-25小时）。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位顶尖的职业规划与学习路径设计师，使用中文回复。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.7,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            time.sleep(0.01)

    except Exception as e:
        yield f"API 调用失败：{str(e)}"

# =====================================
# 主页路由
# =====================================
@app.route('/', methods=['GET', 'POST'])
def index():
    html = """
    <!doctype html>
    <html lang="zh">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>职业学习路径生成器</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; background: #f8f9fa; }
            .card { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            #editor { min-height: 400px; border: 1px solid #ddd; padding: 15px; background: white; border-radius: 8px; }
            .loading { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card p-4 mb-4">
                <h1 class="text-center mb-4">🛤️ 职业学习路径生成器</h1>
                <p class="text-center text-muted">基于 DeepSeek AI，为你定制专业、可执行的学习路线</p>
                
                <form id="generateForm" class="mb-4">
                    <div class="input-group">
                        <input type="text" id="profession" class="form-control form-control-lg" 
                               placeholder="输入职业名称，例如：软件工程师、产品经理、UI设计师..." required>
                        <button type="submit" class="btn btn-primary btn-lg">生成路径</button>
                    </div>
                </form>

                <div class="loading alert alert-info mt-3">
                    <strong>正在生成中...</strong> 请耐心等待 10-30 秒（内容会逐字出现）
                </div>
            </div>

            <div id="resultSection" class="card p-4 d-none">
                <h3>生成结果 <small class="text-muted">(可直接编辑内容)</small></h3>
                <div id="editor" contenteditable="true" class="mb-4"></div>

                <div class="btn-group w-100">
                    <button id="downloadMd" class="btn btn-outline-success">下载 Markdown</button>
                    <button id="downloadExcel" class="btn btn-outline-primary">下载 Excel 表格</button>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script>
            const form = document.getElementById('generateForm');
            const professionInput = document.getElementById('profession');
            const loading = document.querySelector('.loading');
            const resultSection = document.getElementById('resultSection');
            const editor = document.getElementById('editor');
            let fullText = '';

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const profession = professionInput.value.trim();
                if (!profession) return;

                loading.style.display = 'block';
                resultSection.classList.add('d-none');
                editor.innerHTML = '';
                fullText = '';

                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ profession })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    fullText += chunk;
                    editor.innerHTML = marked.parse(fullText + '▌');
                }

                loading.style.display = 'none';
                resultSection.classList.remove('d-none');
                editor.innerHTML = marked.parse(fullText);

                // 下载 Markdown
                document.getElementById('downloadMd').onclick = () => {
                    const blob = new Blob([fullText], { type: 'text/markdown' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `学习路径_${profession}.md`;
                    a.click();
                };

                // 下载 Excel
                document.getElementById('downloadExcel').onclick = () => {
                    const professionEncoded = encodeURIComponent(profession);
                    window.location.href = `/download_excel?profession=${professionEncoded}&content=${encodeURIComponent(fullText)}`;
                };
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

# =====================================
# 流式生成接口
# =====================================
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    profession = data.get('profession', '')

    def event_stream():
        for text in generate_learning_path(profession):
            yield text

    return Response(event_stream(), mimetype='text/plain')

# =====================================
# 下载 Excel 接口（将 Markdown 内容结构化导出为表格）
# =====================================
@app.route('/download_excel')
def download_excel():
    profession = request.args.get('profession', '未知职业')
    raw_content = request.args.get('content', '')

    # 简单解析 Markdown，提取主要部分作为表格数据
    lines = raw_content.split('\n')
    data = []
    current_section = ""
    for line in lines:
        line = line.strip()
        if line.startswith('##') or line.startswith('###') or line.startswith('- **'):
            current_section = line.replace('##', '').replace('###', '').replace('- **', '').replace('**', '').strip()
        elif line.startswith('-') or line.startswith('1.') or line.startswith('|'):
            if line.startswith('|'):
                # 表格行直接添加
                data.append([current_section, "表格数据", line])
            else:
                item = line.lstrip('- 1234567890. ').strip()
                if item:
                    data.append([current_section, item, ""])

    # 创建 DataFrame
    df = pd.DataFrame(data, columns=["模块/部分", "内容项", "备注"])

    # 输出到 Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='学习路径')

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=f"学习路径_{profession}.xlsx",
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# =====================================
# 启动
# =====================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)