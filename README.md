# 使用 Colab 运行 langchain-ChatGLM 项目

https://colab.research.google.com/drive/1o-MJynoh9fW7zGS4IMRnakhy2gE4dVBv?usp=sharing

打开上面的笔记，确认是GPU环境，运行即可，最后会输出一个临时的公网域名，点击即可进入应用。

简单修改了一下 Prompt，感觉效果还不错。

```python
PROMPT_TEMPLATE = """
以下是一些已知的信息：
{context} 

请根据这些信息，以专业且简洁的方式回答下列问题：“{question}”。
在回答问题时，如果某些信息与问题无关，可以选择忽略。如果可以从这些信息中找到答案，请直接回答。如果无法直接从这些信息中找到答案，但可以基于这些信息进行合理的推断，那么请给出推断，并明确指出这是一种推断。
如果这些信息并不足以回答这个问题，那么请回答：“对不起，我无法根据提供的信息回答这个问题。可能需要更多的相关信息。” 请尽量避免在回答中添加任何基于猜测或创作的内容。所有的回答都应该以中文提供。
"""
```


[项目原始的README.md](./README_raw.md)
