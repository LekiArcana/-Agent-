# 基于 Langchain 的多 Agent 法律问答系统

## 概述

这是一个基于 Langchain 框架重构的新一代多 Agent 法律问答系统，实现了智能法律条文检索、专业问答生成、对话总结分析等功能。系统采用模块化的 Agent 架构，支持多轮对话和上下文维护。

## 系统架构

### 核心组件

1. **检索 Agent (Retrieval Agent)**
   - 智能法律条文检索
   - 向量相似度搜索
   - 自适应检索参数调整

2. **问答 Agent (QA Agent)**
   - 专业法律问答生成
   - 多种回答风格支持
   - 上下文感知对话

3. **总结 Agent (Summary Agent)**
   - 对话总结生成
   - 法律内容分析
   - 会话洞察提取

4. **内存管理器 (Memory Manager)**
   - 多会话管理
   - 对话历史维护
   - 上下文自动总结

5. **多 Agent 协调器 (Multi-Agent Coordinator)**
   - Agent 任务调度
   - 执行流程协调
   - 结果整合处理

### 技术栈

- **框架**: Langchain
- **LLM**: Ollama (qwen2.5)
- **向量数据库**: FAISS
- **Embedding**: BGE-Large-ZH-v1.5
- **编程语言**: Python 3.8+

## 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Ollama 服务

确保本地已安装并启动 Ollama 服务，并拉取 qwen2.5 模型：

```bash
# 启动 ollama 服务
ollama serve

# 拉取 qwen2.5 模型
ollama pull qwen2.5
```

### 3. 验证数据文件

确保 `Data` 文件夹下存在以下文件：
- `law_index.faiss` - FAISS 向量索引
- `law_metadata.pkl` - 法律条文元数据

## 使用方法

### 方式一：交互式命令行

```bash
cd qa_system
python langchain_qa_system.py
```

### 方式二：API 调用

```python
import asyncio
from langchain_qa_system import LangchainLegalQASystem

async def main():
    # 创建系统实例
    system = LangchainLegalQASystem()
    
    # 初始化系统
    await system.initialize()
    
    # 提问
    result = await system.ask_question(
        question="什么是合同违约？",
        answer_style="professional"
    )
    
    print(result["answer"])

asyncio.run(main())
```

### 方式三：运行测试和演示

```bash
# 运行完整测试
python test_langchain_system.py

# 运行快速演示
python test_langchain_system.py demo
```

## 功能特性

### 1. 多轮对话支持

系统自动维护对话上下文，支持连续提问和追问：

```
用户: 什么是劳动合同？
助手: [提供劳动合同的定义和说明]

用户: 那么劳动合同应该包含哪些内容？
助手: [基于上下文回答合同内容要求]
```

### 2. 多种回答风格

- **专业风格 (professional)**: 适合法律专业人士
- **简单风格 (simple)**: 通俗易懂，适合普通用户
- **详细风格 (detailed)**: 全面深入的学术分析

### 3. 智能检索

- 自动调整检索参数
- 复杂度分析和适应
- 相关性评估和过滤

### 4. 会话管理

- 多会话并行支持
- 会话历史保存
- 智能会话切换

### 5. 内容分析

- 对话智能总结
- 法律条文分析
- 会话洞察生成

## 交互式命令

在交互模式下，支持以下命令：

| 命令 | 说明 | 示例 |
|------|------|------|
| 直接输入问题 | 进行法律咨询 | `什么是知识产权？` |
| `summary` | 获取对话总结 | `summary` |
| `status` | 查看系统状态 | `status` |
| `new [标题]` | 创建新会话 | `new 公司法咨询` |
| `switch <会话ID>` | 切换会话 | `switch abc123...` |
| `sessions` | 列出所有会话 | `sessions` |
| `style <风格>` | 设置回答风格 | `style simple` |
| `analyze <内容>` | 分析法律内容 | `analyze 第一条 ...` |
| `help` | 查看帮助 | `help` |
| `quit` | 退出系统 | `quit` |

## API 接口

### 主要方法

```python
# 初始化系统
await system.initialize()

# 提问接口
result = await system.ask_question(
    question: str,           # 用户问题
    answer_style: str,       # 回答风格
    show_details: bool       # 显示详细信息
)

# 获取对话总结
summary = await system.get_conversation_summary()

# 分析法律内容
analysis = await system.analyze_content(
    content: str,            # 分析内容
    analysis_type: str       # 分析类型
)

# 批量处理问题
batch_result = await system.batch_questions(
    questions: List[str],    # 问题列表
    answer_style: str        # 回答风格
)

# 会话管理
session_id = system.create_new_session(title)
success = system.switch_session(session_id)
sessions = system.list_sessions()
```

### 返回结果格式

```python
{
    "success": True,           # 是否成功
    "question": "用户问题",    # 原始问题
    "answer": "回答内容",      # 生成的回答
    "answer_style": "风格",    # 回答风格
    "execution_time": 2.5,     # 执行时间
    "session_id": "会话ID",    # 会话标识
    "retrieval_info": {        # 检索信息
        "documents_found": 3,
        "retrieval_analysis": "...",
        "documents": [...]
    },
    "qa_info": {              # 问答信息
        "question_type": "general",
        "follow_up": False,
        "context_used": True
    }
}
```

## 性能指标

基于测试环境的性能表现：

- **平均响应时间**: 3-8 秒
- **检索准确率**: >90%
- **多轮对话支持**: 无限轮次
- **并发会话**: 支持多个独立会话
- **内存使用**: 自动管理，支持长对话

## 故障排除

### 常见问题

1. **系统初始化失败**
   - 检查 Ollama 服务是否启动
   - 确认 qwen2.5 模型已安装
   - 验证数据文件是否存在

2. **检索结果为空**
   - 检查 FAISS 索引文件
   - 确认 BGE 模型正常加载
   - 尝试调整检索参数

3. **回答生成失败**
   - 检查 LLM 连接状态
   - 确认网络连接正常
   - 尝试重启 Ollama 服务

4. **内存占用过高**
   - 限制对话历史长度
   - 定期清理会话数据
   - 调整内存管理参数

### 日志和调试

系统提供详细的执行日志，包括：
- Agent 初始化状态
- 检索执行过程
- 问答生成详情
- 错误信息和堆栈

## 扩展开发

### 添加新的 Agent

1. 继承 `BaseAgent` 类
2. 实现 `initialize()` 和 `execute()` 方法
3. 注册到协调器中

### 自定义工具

1. 使用 `@tool` 装饰器定义新工具
2. 添加到工具列表中
3. 更新 Agent 的工具配置

### 集成其他 LLM

1. 实现新的 LLM 配置类
2. 更新 `llm_config.py`
3. 修改相关配置参数

## 版本历史

- **v2.0.0**: 基于 Langchain 的完全重构版本
  - 多 Agent 架构
  - 改进的内存管理
  - 增强的工具集成
  - 更好的错误处理

- **v1.x**: 原始版本（基于自定义框架）

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进系统：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目遵循 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者
- 加入项目讨论群

---

**感谢使用 Langchain 多 Agent 法律问答系统！** 
