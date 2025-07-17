/**
 * Langchain 多 Agent 法律问答系统 - 前端 JavaScript
 * 
 * 功能模块：
 * - 系统初始化和状态管理
 * - 聊天界面交互
 * - API 通信
 * - 会话管理
 * - 模态框管理
 * - 响应式设计支持
 */

class LegalQAApp {
    constructor() {
        // 应用状态
        this.state = {
            systemReady: false,
            currentSessionId: null,
            isLoading: false,
            autoScroll: true,
            darkMode: false
        };

        // DOM 元素引用
        this.elements = {};
        
        // 初始化应用
        this.initializeApp();
    }

    /**
     * 应用初始化
     */
    async initializeApp() {
        try {
            console.log('🚀 正在初始化 Langchain 法律问答系统...');
            
            // 初始化 DOM 元素
            this.initializeElements();
            
            // 绑定事件监听器
            this.bindEventListeners();
            
            // 检查系统状态
            await this.checkSystemStatus();
            
            // 加载会话列表
            await this.loadSessions();
            
            // 初始化主题
            this.initializeTheme();
            
            console.log('✅ 应用初始化完成');
            
        } catch (error) {
            console.error('❌ 应用初始化失败:', error);
            this.showNotification('系统初始化失败', 'error');
        }
    }

    /**
     * 初始化 DOM 元素引用
     */
    initializeElements() {
        // 基础元素
        this.elements.sidebar = document.getElementById('sidebar');
        this.elements.menuBtn = document.getElementById('menuBtn');
        this.elements.sidebarToggle = document.getElementById('sidebarToggle');
        
        // 系统状态
        this.elements.statusDot = document.getElementById('statusDot');
        this.elements.statusText = document.getElementById('statusText');
        
        // 聊天相关
        this.elements.welcomeScreen = document.getElementById('welcomeScreen');
        this.elements.messagesContainer = document.getElementById('messagesContainer');
        this.elements.messages = document.getElementById('messages');
        this.elements.messageInput = document.getElementById('messageInput');
        this.elements.sendBtn = document.getElementById('sendBtn');
        this.elements.charCount = document.getElementById('charCount');
        this.elements.answerStyle = document.getElementById('answerStyle');
        
        // 会话管理
        this.elements.newChatBtn = document.getElementById('newChatBtn');
        this.elements.sessionsList = document.getElementById('sessionsList');
        
        // 功能按钮
        this.elements.summaryBtn = document.getElementById('summaryBtn');
        this.elements.analyzeBtn = document.getElementById('analyzeBtn');
        this.elements.settingsBtn = document.getElementById('settingsBtn');
        
        // 模态框
        this.elements.summaryModal = document.getElementById('summaryModal');
        this.elements.analyzeModal = document.getElementById('analyzeModal');
        this.elements.settingsModal = document.getElementById('settingsModal');
        
        // 加载和通知
        this.elements.loadingOverlay = document.getElementById('loadingOverlay');
        this.elements.notifications = document.getElementById('notifications');
    }

    /**
     * 绑定事件监听器
     */
    bindEventListeners() {
        // 侧边栏控制
        this.elements.menuBtn?.addEventListener('click', () => this.toggleSidebar());
        this.elements.sidebarToggle?.addEventListener('click', () => this.toggleSidebar());
        
        // 消息输入
        this.elements.messageInput?.addEventListener('input', (e) => this.handleInputChange(e));
        this.elements.messageInput?.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.elements.sendBtn?.addEventListener('click', () => this.sendMessage());
        
        // 会话管理
        this.elements.newChatBtn?.addEventListener('click', () => this.createNewSession());
        
        // 功能按钮
        this.elements.summaryBtn?.addEventListener('click', () => this.showConversationSummary());
        this.elements.analyzeBtn?.addEventListener('click', () => this.showContentAnalyzer());
        this.elements.settingsBtn?.addEventListener('click', () => this.showSettings());
        
        // 示例问题点击
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('question-chip')) {
                const question = e.target.dataset.question;
                if (question) {
                    this.elements.messageInput.value = question;
                    this.handleInputChange({ target: this.elements.messageInput });
                    this.sendMessage();
                }
            }
        });
        
        // 模态框关闭
        this.bindModalEvents();
        
        // 窗口大小变化
        window.addEventListener('resize', () => this.handleResize());
    }

    /**
     * 绑定模态框事件
     */
    bindModalEvents() {
        // 总结模态框
        document.getElementById('summaryModalClose')?.addEventListener('click', () => {
            this.hideModal('summaryModal');
        });
        
        // 分析模态框
        document.getElementById('analyzeModalClose')?.addEventListener('click', () => {
            this.hideModal('analyzeModal');
        });
        
        document.getElementById('analyzeSubmitBtn')?.addEventListener('click', () => {
            this.performContentAnalysis();
        });
        
        // 设置模态框
        document.getElementById('settingsModalClose')?.addEventListener('click', () => {
            this.hideModal('settingsModal');
        });
        
        document.getElementById('darkModeToggle')?.addEventListener('change', (e) => {
            this.toggleDarkMode(e.target.checked);
        });
        
        document.getElementById('autoScrollToggle')?.addEventListener('change', (e) => {
            this.state.autoScroll = e.target.checked;
        });
        
        // 点击模态框背景关闭
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.classList.remove('show');
            }
        });
    }

    /**
     * 检查系统状态
     */
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            if (data.success && data.status === 'ready') {
                this.state.systemReady = true;
                this.updateSystemStatus('connected', '系统就绪');
                console.log('✅ 系统状态：就绪');
            } else {
                this.state.systemReady = false;
                this.updateSystemStatus('error', data.message || '系统未就绪');
                console.warn('⚠️ 系统状态：未就绪');
            }
            
        } catch (error) {
            console.error('❌ 检查系统状态失败:', error);
            this.state.systemReady = false;
            this.updateSystemStatus('error', '连接失败');
        }
    }

    /**
     * 更新系统状态显示
     */
    updateSystemStatus(status, text) {
        const statusDot = this.elements.statusDot;
        const statusText = this.elements.statusText;
        
        if (statusDot) {
            statusDot.className = 'status-dot';
            if (status === 'connected') {
                statusDot.classList.add('connected');
            }
        }
        
        if (statusText) {
            statusText.textContent = text;
        }
    }

    /**
     * 处理输入变化
     */
    handleInputChange(event) {
        const input = event.target;
        const value = input.value;
        const length = value.length;
        
        // 更新字符计数
        if (this.elements.charCount) {
            this.elements.charCount.textContent = length;
        }
        
        // 控制发送按钮状态
        const isEmpty = value.trim().length === 0;
        if (this.elements.sendBtn) {
            this.elements.sendBtn.disabled = isEmpty || !this.state.systemReady;
        }
        
        // 自动调整文本域高度
        this.autoResizeTextarea(input);
    }

    /**
     * 自动调整文本域高度
     */
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 120);
        textarea.style.height = newHeight + 'px';
    }

    /**
     * 处理键盘事件
     */
    handleKeyDown(event) {
        if (event.key === 'Enter') {
            if (event.shiftKey) {
                // Shift+Enter 换行
                return;
            } else {
                // Enter 发送消息
                event.preventDefault();
                if (!this.elements.sendBtn.disabled) {
                    this.sendMessage();
                }
            }
        }
    }

    /**
     * 发送消息
     */
    async sendMessage() {
        const messageText = this.elements.messageInput.value.trim();
        if (!messageText || !this.state.systemReady) return;
        
        try {
            // 禁用输入
            this.setInputState(false);
            
            // 显示用户消息
            this.addMessage('user', messageText);
            
            // 清空输入框
            this.elements.messageInput.value = '';
            this.handleInputChange({ target: this.elements.messageInput });
            
            // 切换到聊天界面
            this.showChatInterface();
            
            // 显示加载指示器
            this.showLoading('AI 正在思考中...');
            
            // 发送 API 请求
            const response = await fetch('/api/question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: messageText,
                    answer_style: this.elements.answerStyle.value,
                    session_id: this.state.currentSessionId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 更新会话 ID
                this.state.currentSessionId = data.data.session_id;
                
                // 显示回答
                this.addMessage('assistant', data.data.answer, {
                    executionTime: data.data.execution_time
                });
                
                // 更新会话列表
                await this.loadSessions();
                
            } else {
                throw new Error(data.message || '问答失败');
            }
            
        } catch (error) {
            console.error('❌ 发送消息失败:', error);
            this.addMessage('assistant', '抱歉，处理您的问题时出现了错误。请稍后重试。', {
                isError: true
            });
            this.showNotification('发送消息失败: ' + error.message, 'error');
            
        } finally {
            // 恢复输入状态
            this.setInputState(true);
            this.hideLoading();
        }
    }

    /**
     * 添加消息到聊天界面
     */
    addMessage(type, content, options = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        // 使用提供的时间戳或当前时间
        const timestamp = options.timestamp ? new Date(options.timestamp) : new Date();
        const timeString = timestamp.toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        const avatarIcon = type === 'user' ? 'fas fa-user' : 'fas fa-robot';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    ${this.formatMessageContent(content)}
                </div>
                <div class="message-time">
                    ${timeString}
                    ${options.executionTime ? ` · 耗时 ${options.executionTime.toFixed(2)}s` : ''}
                    ${options.isError ? ' · 错误' : ''}
                    ${options.isHistorical ? ' · 历史消息' : ''}
                </div>
            </div>
        `;
        
        this.elements.messages.appendChild(messageDiv);
        
        // 自动滚动到底部
        if (this.state.autoScroll) {
            this.scrollToBottom();
        }
    }

    /**
     * 格式化消息内容
     */
    formatMessageContent(content) {
        // 转义 HTML 特殊字符
        const escaped = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
        
        // 处理换行
        return escaped.replace(/\n/g, '<br>');
    }

    /**
     * 滚动到底部
     */
    scrollToBottom() {
        const container = this.elements.messagesContainer;
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    /**
     * 显示聊天界面
     */
    showChatInterface() {
        if (this.elements.welcomeScreen) {
            this.elements.welcomeScreen.style.display = 'none';
        }
        if (this.elements.messagesContainer) {
            this.elements.messagesContainer.style.display = 'block';
        }
    }

    /**
     * 设置输入状态
     */
    setInputState(enabled) {
        if (this.elements.messageInput) {
            this.elements.messageInput.disabled = !enabled;
        }
        if (this.elements.sendBtn) {
            this.elements.sendBtn.disabled = !enabled || !this.state.systemReady;
        }
    }

    /**
     * 创建新会话
     */
    async createNewSession() {
        try {
            const response = await fetch('/api/session/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: `新对话 ${new Date().toLocaleString('zh-CN')}`
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.state.currentSessionId = data.data.session_id;
                
                // 清空当前聊天
                this.elements.messages.innerHTML = '';
                
                // 显示欢迎界面
                this.elements.welcomeScreen.style.display = 'block';
                this.elements.messagesContainer.style.display = 'none';
                
                // 更新会话列表
                await this.loadSessions();
                
                this.showNotification('新对话已创建', 'success');
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('❌ 创建新会话失败:', error);
            this.showNotification('创建新对话失败', 'error');
        }
    }

    /**
     * 加载会话列表
     */
    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            
            if (data.success) {
                this.renderSessionsList(data.data.sessions);
            }
            
        } catch (error) {
            console.error('❌ 加载会话列表失败:', error);
        }
    }

    /**
     * 渲染会话列表
     */
    renderSessionsList(sessions) {
        if (!this.elements.sessionsList) return;
        
        this.elements.sessionsList.innerHTML = '';
        
        sessions.forEach(session => {
            const sessionDiv = document.createElement('div');
            sessionDiv.className = 'session-item';
            if (session.id === this.state.currentSessionId) {
                sessionDiv.classList.add('active');
            }
            
            sessionDiv.innerHTML = `
                <div class="session-title">${session.title || '未命名对话'}</div>
                <div class="session-meta">
                    <span>${session.message_count || 0} 条消息</span>
                    <span>${this.formatDate(session.created_at)}</span>
                </div>
            `;
            
            sessionDiv.addEventListener('click', () => this.switchSession(session.id));
            this.elements.sessionsList.appendChild(sessionDiv);
        });
    }

    /**
     * 切换会话
     */
    async switchSession(sessionId) {
        try {
            const response = await fetch(`/api/session/${sessionId}/switch`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.state.currentSessionId = sessionId;
                
                // 清空当前聊天
                this.elements.messages.innerHTML = '';
                
                // 加载历史消息
                await this.loadSessionMessages(sessionId);
                
                // 更新会话列表显示
                this.renderSessionsList(await this.getSessions());
                
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('❌ 切换会话失败:', error);
            this.showNotification('切换会话失败', 'error');
        }
    }

    /**
     * 加载指定会话的历史消息
     */
    async loadSessionMessages(sessionId) {
        try {
            const response = await fetch(`/api/session/${sessionId}/messages`);
            const data = await response.json();
            
            if (data.success) {
                const messages = data.data.messages;
                
                if (messages && messages.length > 0) {
                    // 显示聊天界面
                    this.elements.welcomeScreen.style.display = 'none';
                    this.elements.messagesContainer.style.display = 'block';
                    
                    // 清空消息容器
                    this.elements.messages.innerHTML = '';
                    
                    // 逐个添加历史消息
                    for (const message of messages) {
                        this.addMessage(message.type, message.content, {
                            timestamp: message.timestamp,
                            isHistorical: true
                        });
                    }
                    
                    // 滚动到底部
                    this.scrollToBottom();
                    
                } else {
                    // 没有历史消息，显示欢迎界面
                    this.elements.welcomeScreen.style.display = 'block';
                    this.elements.messagesContainer.style.display = 'none';
                }
                
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('❌ 加载历史消息失败:', error);
            // 如果加载失败，显示欢迎界面
            this.elements.welcomeScreen.style.display = 'block';
            this.elements.messagesContainer.style.display = 'none';
        }
    }

    /**
     * 获取会话列表
     */
    async getSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            return data.success ? data.data.sessions : [];
        } catch {
            return [];
        }
    }

    /**
     * 显示对话总结
     */
    async showConversationSummary() {
        this.showModal('summaryModal');
        
        const summaryContent = document.getElementById('summaryContent');
        if (!summaryContent) return;
        
        summaryContent.innerHTML = `
            <div class="loading">
                <i class="fas fa-spinner fa-spin"></i>
                <span>正在生成总结...</span>
            </div>
        `;
        
        try {
            const response = await fetch('/api/conversation/summary');
            const data = await response.json();
            
            if (data.success) {
                summaryContent.innerHTML = `
                    <div class="summary-result">
                        ${this.formatMessageContent(data.data.summary || '暂无对话内容可以总结')}
                    </div>
                `;
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('❌ 获取对话总结失败:', error);
            summaryContent.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>获取对话总结失败: ${error.message}</span>
                </div>
            `;
        }
    }

    /**
     * 显示内容分析器
     */
    showContentAnalyzer() {
        this.showModal('analyzeModal');
        
        // 重置分析结果
        const analyzeResult = document.getElementById('analyzeResult');
        if (analyzeResult) {
            analyzeResult.style.display = 'none';
            analyzeResult.innerHTML = '';
        }
    }

    /**
     * 执行内容分析
     */
    async performContentAnalysis() {
        const textarea = document.getElementById('analyzeTextarea');
        const analysisType = document.getElementById('analysisType');
        const resultDiv = document.getElementById('analyzeResult');
        
        if (!textarea || !analysisType || !resultDiv) return;
        
        const content = textarea.value.trim();
        if (!content) {
            this.showNotification('请输入要分析的内容', 'warning');
            return;
        }
        
        try {
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `
                <div class="loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span>正在分析内容...</span>
                </div>
            `;
            
            const response = await fetch('/api/content/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: content,
                    analysis_type: analysisType.value
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                resultDiv.innerHTML = `
                    <h4>分析结果</h4>
                    <div class="analysis-content">
                        ${this.formatMessageContent(data.data.analysis || '分析完成')}
                    </div>
                `;
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('❌ 内容分析失败:', error);
            resultDiv.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>内容分析失败: ${error.message}</span>
                </div>
            `;
        }
    }

    /**
     * 显示设置
     */
    async showSettings() {
        this.showModal('settingsModal');
        
        // 加载系统信息
        await this.loadSystemInfo();
    }

    /**
     * 加载系统信息
     */
    async loadSystemInfo() {
        const systemInfo = document.getElementById('systemInfo');
        if (!systemInfo) return;
        
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            if (data.success && data.data) {
                const info = data.data;
                systemInfo.innerHTML = `
                    <div class="info-item">
                        <div class="info-label">LLM 模型</div>
                        <div class="info-value">${info.llm_model || 'Unknown'}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Embedding 模型</div>
                        <div class="info-value">${info.embedding_model || 'Unknown'}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">向量数据库</div>
                        <div class="info-value">${info.vector_count || 0} 个向量</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">工具数量</div>
                        <div class="info-value">${info.tools_count || 0} 个工具</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Agent 状态</div>
                        <div class="info-value">
                            ${info.agents_status ? '✅ 正常' : '❌ 异常'}
                        </div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">启动时间</div>
                        <div class="info-value">${this.formatDate(info.startup_time)}</div>
                    </div>
                `;
            }
            
        } catch (error) {
            console.error('❌ 加载系统信息失败:', error);
            systemInfo.innerHTML = `
                <div class="error-message">
                    <span>加载系统信息失败</span>
                </div>
            `;
        }
    }

    /**
     * 显示模态框
     */
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('show');
        }
    }

    /**
     * 隐藏模态框
     */
    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('show');
        }
    }

    /**
     * 切换侧边栏
     */
    toggleSidebar() {
        if (this.elements.sidebar) {
            this.elements.sidebar.classList.toggle('open');
        }
    }

    /**
     * 初始化主题
     */
    initializeTheme() {
        const darkModeToggle = document.getElementById('darkModeToggle');
        const savedTheme = localStorage.getItem('theme');
        
        if (savedTheme === 'dark') {
            this.state.darkMode = true;
            document.documentElement.setAttribute('data-theme', 'dark');
            if (darkModeToggle) {
                darkModeToggle.checked = true;
            }
        }
    }

    /**
     * 切换深色模式
     */
    toggleDarkMode(enabled) {
        this.state.darkMode = enabled;
        
        if (enabled) {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
            localStorage.setItem('theme', 'light');
        }
    }

    /**
     * 显示加载指示器
     */
    showLoading(text = '加载中...') {
        if (this.elements.loadingOverlay) {
            const loadingText = this.elements.loadingOverlay.querySelector('.loading-text');
            if (loadingText) {
                loadingText.textContent = text;
            }
            this.elements.loadingOverlay.style.display = 'flex';
        }
    }

    /**
     * 隐藏加载指示器
     */
    hideLoading() {
        if (this.elements.loadingOverlay) {
            this.elements.loadingOverlay.style.display = 'none';
        }
    }

    /**
     * 显示通知
     */
    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        const titles = {
            success: '成功',
            error: '错误',
            warning: '警告',
            info: '信息'
        };
        
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-icon">
                    <i class="${icons[type] || icons.info}"></i>
                </div>
                <div class="notification-text">
                    <div class="notification-title">${titles[type] || titles.info}</div>
                    <div class="notification-message">${message}</div>
                </div>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // 添加关闭事件
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            notification.remove();
        });
        
        // 添加到容器
        this.elements.notifications.appendChild(notification);
        
        // 自动移除
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, duration);
    }

    /**
     * 格式化日期
     */
    formatDate(dateString) {
        if (!dateString) return '';
        
        try {
            const date = new Date(dateString);
            const now = new Date();
            const diffDays = Math.floor((now - date) / (1000 * 60 * 60 * 24));
            
            if (diffDays === 0) {
                return '今天';
            } else if (diffDays === 1) {
                return '昨天';
            } else if (diffDays < 7) {
                return `${diffDays} 天前`;
            } else {
                return date.toLocaleDateString('zh-CN');
            }
        } catch {
            return '';
        }
    }

    /**
     * 处理窗口大小变化
     */
    handleResize() {
        // 移动端适配
        if (window.innerWidth <= 768) {
            // 关闭侧边栏
            if (this.elements.sidebar) {
                this.elements.sidebar.classList.remove('open');
            }
        }
    }
}

// 应用启动
document.addEventListener('DOMContentLoaded', () => {
    window.legalQAApp = new LegalQAApp();
});

// 错误处理
window.addEventListener('error', (event) => {
    console.error('全局错误:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的 Promise 拒绝:', event.reason);
}); 