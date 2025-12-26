import os
import pickle

import seaborn as sns
import streamlit as st
import time
import glob
from PIL import Image

# 导入所有已提供的类和函数
from DQN_extract import *

# [这里导入所有其他必要的类和函数，如ErrorInjector, DataCleaningEnv, DQNAgent等]

# 设置页面配置
st.set_page_config(
    page_title="CBTAD - Cost-Based Tolerance Analysis for Data Quality",
    page_icon="📊",
    layout="wide"
)


def save_model_step2(agent, model_type, task_type, error_rate):
    """保存模型和训练历史"""
    # 保存模型
    # model_filename = f"dqn_agent_{task_type}_{model_type}_{error_rate:.2f}.h5"
    model_filename = f"dqn_agent_{task_type}_{model_type}.pt"
    agent.save(model_filename)

    # 保存训练历史
    history = {
        'epsilon_history': st.session_state.get('epsilon_history', []),
        'q_values_history': st.session_state.get('q_values_history', []),
        'reward_history': st.session_state.get('reward_history', [])
    }

    # history_filename = f"dqn_history_{task_type}_{model_type}_{error_rate:.2f}.pkl"
    history_filename = f"dqn_history_{task_type}_{model_type}.pkl"
    with open(history_filename, 'wb') as f:
        pickle.dump(history, f)

    return model_filename


def load_model_step2(model_type, task_type, error_rate):
    """加载模型和训练历史"""
    # model_filename = f"dqn_agent_{task_type}_{model_type}_{error_rate:.2f}.h5"
    # history_filename = f"dqn_history_{task_type}_{model_type}_{error_rate:.2f}.pkl"
    model_filename = f"dqn_agent_{task_type}_{model_type}.pt"
    history_filename = f"dqn_history_{task_type}_{model_type}.pkl"
    agent = None
    history = None

    try:
        # 尝试加载模型
        if os.path.exists(model_filename):
            agent = DQNAgent(state_size=5, action_size=3)
            agent.load(model_filename)

        # 尝试加载历史数据
        if os.path.exists(history_filename):
            with open(history_filename, 'rb') as f:
                history = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model or history: {e}")

    return agent, history


# 检查模型是否已存在
def check_model_exists(model_name, task_type, error_rate):
    """检查特定模型、任务类型和错误率的模型文件是否存在"""
    # filename = f"dqn_agent_{task_type}_{model_name}_{error_rate:.2f}.h5"
    filename = f"dqn_agent_{task_type}_{model_name}.pt"
    return os.path.exists(filename)


#
# 保存模型
def save_model(agent, model_name, task_type, error_rate):
    """保存模型到指定文件名"""
    # filename = f"dqn_agent_{task_type}_{model_name}_{error_rate:.2f}.h5"
    filename = f"dqn_agent_{task_type}_{model_name}.pt"
    agent.save(filename)
    return filename


# 加载模型
def load_model(model_name, task_type, error_rate):
    """加载模型"""
    # filename = f"dqn_agent_{task_type}_{model_name}_{error_rate:.2f}.h5"
    filename = f"dqn_agent_{task_type}_{model_name}.pt"
    if os.path.exists(filename):
        agent = DQNAgent(state_size=5, action_size=3)
        agent.load(filename)
        return agent
    return None


# 查找相关的图片文件
def find_images(task_type, error_rate, prefix=""):
    """查找与特定任务类型和错误率相关的图片文件"""
    pattern = f"{prefix}*{task_type}*{error_rate:.2f}*.png"
    return glob.glob(pattern)


# 侧边栏导航
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Step 1: Data Upload & Configuration",
                                  "Step 2: RL Training",
                                  "Step 3: Strategy Comparison",
                                  "Step 4: Model Comparison",
                                  "Step 5: Export Results"])

# 标题
st.title("CBTAD - Cost-Based Tolerance Analysis for Data Quality")

if page == "Step 1: Data Upload & Configuration":
    st.header("Step 1: Data Upload & Parameter Configuration")

    # 数据上传部分
    st.subheader("Upload Data or Use Sample")
    data_option = st.radio("Choose data source:", ["Upload my own data", "Use sample dataset"])

    if data_option == "Upload my own data":
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['clean_df'] = df
                st.success(f"Successfully uploaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    else:
        task_type = st.radio("Choose task type for sample data:", ["classification", "regression"])
        n_samples = st.slider("Number of samples:", min_value=100, max_value=5000, value=1000, step=100)
        n_features = st.slider("Number of features:", min_value=2, max_value=20, value=5, step=1)

        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                df = generate_clean_data(task_type=task_type, n_samples=n_samples, n_features=n_features)
                st.session_state['clean_df'] = df
                st.session_state['task_type'] = task_type
                st.success(f"Generated {task_type} dataset with {n_samples} samples and {n_features} features.")

    # 显示数据预览
    if 'clean_df' in st.session_state:
        st.subheader("Data Preview")
        st.dataframe(st.session_state['clean_df'].head(10))

        # 数据可视化
        # 优化可视化数据展示

    # 错误注入参数
    st.subheader("Error Injection Parameters")

    error_rate = st.slider("Total error rate:",
                           min_value=0.01, max_value=1.0, value=0.3, step=0.01)

    col1, col2, col3 = st.columns(3)
    with col1:
        missing_ratio = st.slider("Missing values ratio:",
                                  min_value=0.0, max_value=1.0, value=0.33, step=0.01)
    with col2:
        outlier_ratio = st.slider("Outlier values ratio:",
                                  min_value=0.0, max_value=1.0, value=0.33, step=0.01)
    with col3:
        noise_ratio = st.slider("Noise ratio:",
                                min_value=0.0, max_value=1.0, value=0.34, step=0.01)

    # 归一化比例总和为1
    total = missing_ratio + outlier_ratio + noise_ratio
    missing_ratio = missing_ratio / total
    outlier_ratio = outlier_ratio / total
    noise_ratio = noise_ratio / total

    st.info(f"Normalized ratios: Missing={missing_ratio:.2f}, Outliers={outlier_ratio:.2f}, Noise={noise_ratio:.2f}")

    # 特征重要性计算
    st.subheader("Feature Importance")
    auto_importance = st.checkbox("Automatically calculate feature importance", value=True)

    if 'clean_df' in st.session_state:
        df = st.session_state['clean_df']
        feature_cols = [col for col in df.columns if col != 'target']

        if auto_importance:
            # 使用默认值，直到计算完成
            importance_values = {col: 1.0 / len(feature_cols) for col in feature_cols}
            st.write("Feature importance will be calculated during training.")
        else:
            st.write("Set importance values manually (values between 0 and 1):")
            importance_values = {}
            for col in feature_cols:
                importance_values[col] = st.slider(f"Importance of {col}",
                                                   min_value=0.0, max_value=1.0, value=0.5, step=0.1)

        # 存储设置到session_state
        st.session_state['error_rate'] = error_rate
        st.session_state['missing_ratio'] = missing_ratio
        st.session_state['outlier_ratio'] = outlier_ratio
        st.session_state['noise_ratio'] = noise_ratio
        st.session_state['feature_importance'] = importance_values
        st.session_state['auto_importance'] = auto_importance

    # 下游模型选择
    st.subheader("Downstream Model Selection")
    task_type = st.session_state.get('task_type', 'classification')

    if task_type == 'classification':
        model_options = ["random_forest", "svm", "logistic_regression"]
    else:
        model_options = ["random_forest", "svm", "linear_regression"]

    model_type = st.selectbox("Select downstream model type:", model_options)
    st.session_state['model_type'] = model_type

    # 注入错误按钮
    if 'clean_df' in st.session_state:
        if st.button("Inject Errors & Proceed to Training"):
            with st.spinner("Injecting errors..."):
                # 创建错误注入器
                injector = ErrorInjector(st.session_state['clean_df'])

                # 如果自动计算，则计算特征重要性
                if st.session_state['auto_importance']:
                    # 使用临时环境计算特征重要性
                    temp_df = injector.inject_missing_values(error_rate=0.01)
                    if task_type == 'classification':
                        ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
                    else:
                        ml_model = RandomForestRegressor(n_estimators=10, random_state=42)

                    temp_env = DataCleaningEnv(temp_df, ml_model, injector.error_locations,
                                               task_type=task_type, model_type=model_type)
                    st.session_state['feature_importance'] = temp_env._calculate_feature_importance()

                # 重置并注入错误
                injector.reset()
                missing_err_rate = error_rate * missing_ratio
                outlier_err_rate = error_rate * outlier_ratio
                noise_err_rate = error_rate * noise_ratio

                df_with_errors = injector.inject_missing_values(error_rate=missing_err_rate,
                                                                feature_importance=st.session_state[
                                                                    'feature_importance'])
                df_with_errors = injector.inject_outliers(error_rate=outlier_err_rate,
                                                          feature_importance=st.session_state['feature_importance'])
                df_with_errors = injector.inject_noise(error_rate=noise_err_rate,
                                                       feature_importance=st.session_state['feature_importance'])

                st.session_state['df_with_errors'] = df_with_errors
                st.session_state['injector'] = injector

                # 查找是否有已保存的错误数据可视化图片
                error_data_images = find_images(task_type, error_rate, f"Injected Errors")
                if error_data_images:
                    st.image(error_data_images[0], caption=f"Data with Injected Errors (Rate: {error_rate:.2f})")
                else:
                    visualize_data(df_with_errors, task_type, f'Injected Errors ({error_rate:.2f})')
                    # 尝试加载刚刚生成的图片
                    error_data_images = find_images(task_type, error_rate, f"Injected Errors")
                    if error_data_images:
                        st.image(error_data_images[0], caption=f"Data with Injected Errors (Rate: {error_rate:.2f})")

                st.session_state['step1_complete'] = True

        if st.session_state.get('step1_complete', True):
            st.success("Errors injected successfully! You can now proceed to training.")
        if st.checkbox("Show data visualization"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Clean Data")
                # 查找是否有已保存的可视化图片
                clean_data_images = find_images(st.session_state.get('task_type', 'classification'), 0.0, "Clean Data")
                if clean_data_images:
                    st.image(clean_data_images[0], use_container_width=True)
                else:
                    fig = visualize_data(st.session_state['clean_df'],
                                         st.session_state.get('task_type', 'classification'),
                                         'Clean Data', return_fig=True)
                    st.pyplot(fig)
                    # 加载刚刚生成的图片
                    clean_data_images = find_images(st.session_state.get('task_type', 'classification'), 0.0,
                                                    "Clean Data")
                    if clean_data_images:
                        st.image(clean_data_images[0], use_container_width=True)

            # 如果已经有错误注入的数据，显示在右侧
            with col2:
                if 'df_with_errors' in st.session_state:
                    st.subheader(f"Data with Errors (Rate: {st.session_state['error_rate']:.2f})")
                    error_data_images = find_images(task_type, st.session_state['error_rate'], f"Injected Errors")
                    if error_data_images:
                        st.image(error_data_images[0], use_container_width=True)
                    else:
                        fig = visualize_data(st.session_state['df_with_errors'], task_type,
                                             f"Injected Errors ({st.session_state['error_rate']:.2f})", return_fig=True)
                        st.pyplot(fig)

elif page == "Step 2: RL Training":
    st.header("Step 2: RL Training Visualization")

    if not st.session_state.get('step1_complete', False):
        st.warning("Please complete Step 1 first.")
        if st.button("Go to Step 1"):
            st.rerun()
    else:
        # 训练参数
        st.subheader("Training Parameters")
        n_episodes = st.slider("Number of training episodes:",
                               min_value=10, max_value=500, value=100, step=10)
        batch_size = st.slider("Batch size for training:",
                               min_value=16, max_value=128, value=32, step=8)

        # 获取模型相关信息
        task_type = st.session_state.get('task_type', 'classification')
        model_type = st.session_state.get('model_type', 'random_forest')
        error_rate = st.session_state['error_rate']

        # 保存和加载模型的函数

        # 检查是否已有训练好的模型
        model_exists = check_model_exists(model_type, task_type, error_rate)

        if model_exists:
            st.info(
                f"A pre-trained model for {model_type} with error rate {error_rate:.2f} exists. You can use it or train a new one.")
            use_existing = st.radio("Use existing model or train new?", ["Use existing", "Train new"])
        else:
            use_existing = "Train new"

        # 初始化RL环境
        if st.button("Start Training"):
            with st.spinner("Initializing environment..."):
                # 从session_state获取数据和设置
                df_with_errors = st.session_state['df_with_errors']
                injector = st.session_state['injector']

                # 为环境初始化ML模型
                if task_type == 'classification':
                    ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
                else:
                    ml_model = RandomForestRegressor(n_estimators=10, random_state=42)

                # 创建环境
                env = DataCleaningEnv(df_with_errors, ml_model, injector.error_locations,
                                      task_type=task_type, model_type=model_type)

                # 创建或加载代理
                if use_existing == "Use existing":
                    # 加载已有模型和历史
                    agent, history = load_model_step2(model_type, task_type, error_rate)

                    if agent:
                        st.success(f"Loaded pre-trained model for {model_type} with error rate {error_rate:.2f}")

                        # 创建指标占位符
                        metrics_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        training_chart = st.empty()

                        # 检查是否有历史数据
                        if history and all(
                                len(history.get(k, [])) > 0 for k in
                                ['epsilon_history', 'q_values_history', 'reward_history']):
                            # 使用加载的历史数据
                            st.info("Using real training history data")
                            epsilon_history = history['epsilon_history']
                            q_values_history = history['q_values_history']
                            reward_history = history['reward_history']

                            # 显示加载的历史数据，为了有动画效果，逐步显示
                            for i in range(1, len(epsilon_history) + 1, max(1, len(epsilon_history) // 50)):
                                # 更新进度
                                progress = i / len(epsilon_history)
                                progress_bar.progress(progress)

                                # 更新指标显示
                                metrics_placeholder.text(f"Loaded Episode: {i}/{len(epsilon_history)} | "
                                                         f"Epsilon: {epsilon_history[i - 1]:.4f} | "
                                                         f"Avg Q-Value: {q_values_history[i - 1]:.4f} | "
                                                         f"Reward: {reward_history[i - 1]:.4f}")

                                # 更新图表
                                if i % max(1, len(epsilon_history) // 10) == 0 or i == len(epsilon_history):
                                    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

                                    # 绘制真实的epsilon历史
                                    axs[0].plot(epsilon_history[:i], color='#1f77b4', linewidth=2)
                                    axs[0].set_title('Epsilon Decay')
                                    axs[0].set_xlabel('Episode')
                                    axs[0].set_ylabel('Epsilon')
                                    axs[0].grid(True, linestyle='--', alpha=0.7)

                                    # 绘制真实的Q值历史
                                    axs[1].plot(q_values_history[:i], color='#ff7f0e', linewidth=2)
                                    axs[1].set_title('Average Q-Values')
                                    axs[1].set_xlabel('Episode')
                                    axs[1].set_ylabel('Q-Value')
                                    axs[1].grid(True, linestyle='--', alpha=0.7)

                                    # 绘制真实的奖励历史
                                    axs[2].plot(reward_history[:i], color='#2ca02c', linewidth=2)
                                    axs[2].set_title('Episode Rewards')
                                    axs[2].set_xlabel('Episode')
                                    axs[2].set_ylabel('Reward')
                                    axs[2].grid(True, linestyle='--', alpha=0.7)

                                    plt.tight_layout()
                                    training_chart.pyplot(fig)
                                    plt.close(fig)

                                time.sleep(0.05)

                            # 将历史数据存储到session_state
                            st.session_state['epsilon_history'] = epsilon_history
                            st.session_state['q_values_history'] = q_values_history
                            st.session_state['reward_history'] = reward_history
                        else:
                            # 如果没有历史数据，重训
                            st.warning("No training history found. Using simulated data for visualization.")

                            epsilon_history = []
                            q_values_history = []
                            reward_history = []

                            # 训练指标
                            for e in range(n_episodes):
                                # 生成真实的数据
                                epsilon = max(0.01, 1.0 * (0.995 ** e))
                                q_value = min(0.8, 0.1 + e * 0.7 / n_episodes) + np.random.normal(0, 0.05)
                                reward = min(0.9, 0.2 + e * 0.7 / n_episodes) + np.random.normal(0, 0.1)

                                epsilon_history.append(epsilon)
                                q_values_history.append(max(0, q_value))  # 确保Q值非负
                                reward_history.append(reward)

                                # 更新进度
                                progress = (e + 1) / n_episodes
                                progress_bar.progress(progress)

                                # 更新指标显示
                                metrics_placeholder.text(f"Simulated Episode: {e + 1}/{n_episodes} | "
                                                         f"Epsilon: {epsilon:.4f} | "
                                                         f"Avg Q-Value: {q_value:.4f} | "
                                                         f"Reward: {reward:.4f}")

                                # 更新图表
                                if (e + 1) % 5 == 0 or e == n_episodes - 1:
                                    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

                                    # 绘制epsilon
                                    axs[0].plot(epsilon_history, color='#1f77b4', linewidth=2)
                                    axs[0].set_title('Epsilon Decay (Simulated)')
                                    axs[0].set_xlabel('Episode')
                                    axs[0].set_ylabel('Epsilon')
                                    axs[0].grid(True, linestyle='--', alpha=0.7)

                                    # 绘制Q值
                                    axs[1].plot(q_values_history, color='#ff7f0e', linewidth=2)
                                    axs[1].set_title('Average Q-Values (Simulated)')
                                    axs[1].set_xlabel('Episode')
                                    axs[1].set_ylabel('Q-Value')
                                    axs[1].grid(True, linestyle='--', alpha=0.7)

                                    # 绘制奖励
                                    axs[2].plot(reward_history, color='#2ca02c', linewidth=2)
                                    axs[2].set_title('Episode Rewards (Simulated)')
                                    axs[2].set_xlabel('Episode')
                                    axs[2].set_ylabel('Reward')
                                    axs[2].grid(True, linestyle='--', alpha=0.7)

                                    plt.tight_layout()
                                    training_chart.pyplot(fig)
                                    plt.close(fig)

                                time.sleep(0.05)

                            # 将数据存储到session_state
                            st.session_state['epsilon_history'] = epsilon_history
                            st.session_state['q_values_history'] = q_values_history
                            st.session_state['reward_history'] = reward_history

                if agent is None:
                    st.warning("No pre-trained agent found, training a new one instead.")

                    use_existing = "Train new"

                if use_existing != "Use existing":  # Train new
                    # 创建新代理
                    agent = DQNAgent(state_size=5, action_size=3)

                    # 创建训练指标的占位符
                    st.session_state['epsilon_history'] = []
                    st.session_state['q_values_history'] = []
                    st.session_state['reward_history'] = []

                    metrics_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    training_chart = st.empty()

                    # 训练代理
                    for e in range(n_episodes):
                        state = env.reset()
                        total_reward = 0
                        episode_q_values = []

                        for _ in range(len(injector.error_locations)):
                            action = agent.act(state)
                            q_values = agent.predict_values(state)
                            episode_q_values.append(np.max(q_values))

                            next_state, reward, done, _ = env.step(action)
                            agent.remember(state, action, reward, next_state, done)
                            state = next_state
                            total_reward += reward

                            if done:
                                break

                        # 基于批次训练
                        agent.replay(batch_size)

                        # 更新历史记录
                        st.session_state['epsilon_history'].append(agent.epsilon)
                        st.session_state['q_values_history'].append(
                            np.mean(episode_q_values) if episode_q_values else 0)
                        st.session_state['reward_history'].append(total_reward)

                        # 更新进度
                        progress_bar.progress((e + 1) / n_episodes)

                        # 更新指标显示
                        metrics_placeholder.text(f"Episode: {e + 1}/{n_episodes} | "
                                                 f"Epsilon: {agent.epsilon:.4f} | "
                                                 f"Avg Q-Value: {np.mean(episode_q_values) if episode_q_values else 0:.4f} | "
                                                 f"Reward: {total_reward:.4f}")

                        # 每5轮更新一次图表
                        if (e + 1) % 5 == 0 or e == n_episodes - 1:
                            fig, axs = plt.subplots(1, 3, figsize=(12, 3))

                            # 绘制epsilon
                            axs[0].plot(st.session_state['epsilon_history'], color='#1f77b4', linewidth=2)
                            axs[0].set_title('Epsilon Decay')
                            axs[0].set_xlabel('Episode')
                            axs[0].set_ylabel('Epsilon')
                            axs[0].grid(True, linestyle='--', alpha=0.7)

                            # 绘制Q值
                            axs[1].plot(st.session_state['q_values_history'], color='#ff7f0e', linewidth=2)
                            axs[1].set_title('Average Q-Values')
                            axs[1].set_xlabel('Episode')
                            axs[1].set_ylabel('Q-Value')
                            axs[1].grid(True, linestyle='--', alpha=0.7)

                            # 绘制奖励
                            axs[2].plot(st.session_state['reward_history'], color='#2ca02c', linewidth=2)
                            axs[2].set_title('Episode Rewards')
                            axs[2].set_xlabel('Episode')
                            axs[2].set_ylabel('Reward')
                            axs[2].grid(True, linestyle='--', alpha=0.7)

                            plt.tight_layout()
                            training_chart.pyplot(fig)
                            plt.close(fig)

                    # 保存训练好的模型和历史
                    model_path = save_model_step2(agent, model_type, task_type, error_rate)
                    st.success(f"Model trained and saved to {model_path}")

                # 将训练好的代理和环境存储到session_state
                st.session_state['env'] = env
                st.session_state['agent'] = agent
                st.session_state['step2_complete'] = True

                st.success("Training complete! You can now proceed to strategy comparison.")

        # 如果训练完成，显示训练结果
        if st.session_state.get('step2_complete', False):
            st.subheader("Training Results")

            if 'epsilon_history' in st.session_state:
                # 创建美观的训练结果可视化
                fig = plt.figure(figsize=(14, 4))

                # 创建3个并排的子图
                gs = fig.add_gridspec(1, 3, hspace=0, wspace=0.3)
                axs = gs.subplots()

                # 绘制epsilon
                axs[0].plot(st.session_state['epsilon_history'], color='#1f77b4', linewidth=2)
                axs[0].set_title('Epsilon Decay')
                axs[0].set_xlabel('Episode')
                axs[0].set_ylabel('Epsilon')
                axs[0].grid(True, linestyle='--', alpha=0.7)

                # 填充区域，使图表更美观
                axs[0].fill_between(range(len(st.session_state['epsilon_history'])),
                                    st.session_state['epsilon_history'],
                                    alpha=0.2, color='#1f77b4')

                # 绘制Q值
                axs[1].plot(st.session_state['q_values_history'], color='#ff7f0e', linewidth=2)
                axs[1].set_title('Average Q-Values')
                axs[1].set_xlabel('Episode')
                axs[1].set_ylabel('Q-Value')
                axs[1].grid(True, linestyle='--', alpha=0.7)
                axs[1].fill_between(range(len(st.session_state['q_values_history'])),
                                    st.session_state['q_values_history'],
                                    alpha=0.2, color='#ff7f0e')

                # 绘制奖励
                axs[2].plot(st.session_state['reward_history'], color='#2ca02c', linewidth=2)
                axs[2].set_title('Episode Rewards')
                axs[2].set_xlabel('Episode')
                axs[2].set_ylabel('Reward')
                axs[2].grid(True, linestyle='--', alpha=0.7)
                axs[2].fill_between(range(len(st.session_state['reward_history'])),
                                    st.session_state['reward_history'],
                                    alpha=0.2, color='#2ca02c')

                # 优化布局
                plt.tight_layout()
                st.pyplot(fig)

                # 添加训练结果的简要统计
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Final Epsilon",
                        f"{st.session_state['epsilon_history'][-1]:.4f}",
                        f"{st.session_state['epsilon_history'][-1] - st.session_state['epsilon_history'][0]:.4f}"
                    )

                with col2:
                    avg_q = np.mean(st.session_state['q_values_history'][-10:])
                    initial_avg_q = np.mean(st.session_state['q_values_history'][:10])
                    st.metric(
                        "Average Q-Value",
                        f"{avg_q:.4f}",
                        f"{avg_q - initial_avg_q:.4f}"
                    )

                with col3:
                    avg_reward = np.mean(st.session_state['reward_history'][-10:])
                    initial_avg_reward = np.mean(st.session_state['reward_history'][:10])
                    st.metric(
                        "Average Reward",
                        f"{avg_reward:.4f}",
                        f"{avg_reward - initial_avg_reward:.4f}"
                    )

elif page == "Step 3: Strategy Comparison":
    st.header("Step 3: Strategy Comparison & Action Statistics")

    if not st.session_state.get('step2_complete', False):
        st.warning("Please complete Step 2 first.")
        if st.button("Go to Step 2"):
            st.rerun()
    else:
        # 添加选项让用户选择数据来源
        data_source = st.radio(
            "Select data source for strategy comparison:",
            ["Use data from previous steps", "Upload new test data", "Generate example test data"]
        )

        # 初始化标志，控制是否执行策略比较
        run_comparison = False
        df_with_errors = None
        clean_df = None
        error_locations = None
        injector = None

        # 根据数据源选择执行不同的初始化
        if data_source == "Use data from previous steps":
            # 使用之前步骤的数据
            if st.button("Run Comparison with Existing Data"):
                df_with_errors = st.session_state['df_with_errors']
                clean_df = st.session_state['clean_df']
                injector = st.session_state['injector']
                error_locations = injector.error_locations
                run_comparison = True

        elif data_source == "Upload new test data":
            st.subheader("Upload Test Data")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Upload data with errors:")
                uploaded_test_file = st.file_uploader("Upload CSV or Excel file with errors", type=["csv", "xlsx"],
                                                      key="test_data")

                if uploaded_test_file:
                    try:
                        if uploaded_test_file.name.endswith('.csv'):
                            test_df_with_errors = pd.read_csv(uploaded_test_file)
                        else:
                            test_df_with_errors = pd.read_excel(uploaded_test_file)
                        st.success(
                            f"Successfully uploaded test data with {test_df_with_errors.shape[0]} rows and {test_df_with_errors.shape[1]} columns.")
                        st.dataframe(test_df_with_errors.head(5))
                    except Exception as e:
                        st.error(f"Error reading test file: {e}")

            with col2:
                st.write("Upload corresponding clean data (for evaluation):")
                uploaded_clean_file = st.file_uploader("Upload CSV or Excel file with clean data", type=["csv", "xlsx"],
                                                       key="clean_data")

                if uploaded_clean_file:
                    try:
                        if uploaded_clean_file.name.endswith('.csv'):
                            test_clean_df = pd.read_csv(uploaded_clean_file)
                        else:
                            test_clean_df = pd.read_excel(uploaded_clean_file)
                        st.success(
                            f"Successfully uploaded clean data with {test_clean_df.shape[0]} rows and {test_clean_df.shape[1]} columns.")
                        st.dataframe(test_clean_df.head(5))
                    except Exception as e:
                        st.error(f"Error reading clean file: {e}")

            # 检测错误位置按钮
            if 'uploaded_test_file' in locals() and 'uploaded_clean_file' in locals():
                if st.button("Detect Error Locations"):
                    # 比较两个数据集找出错误位置
                    error_locations = []
                    if test_df_with_errors.shape == test_clean_df.shape:
                        for i in range(len(test_df_with_errors)):
                            for col in test_df_with_errors.columns:
                                if col != 'target' and pd.notna(test_df_with_errors.loc[i, col]) and pd.notna(
                                        test_clean_df.loc[i, col]):
                                    if test_df_with_errors.loc[i, col] != test_clean_df.loc[i, col]:
                                        # 检查是否为缺失值或异常值
                                        error_type = 0  # 默认为缺失值
                                        if pd.isna(test_df_with_errors.loc[i, col]):
                                            error_type = 0  # 缺失值
                                        else:
                                            # 简单异常值检测 (可以根据需要调整)
                                            col_std = test_clean_df[col].std()
                                            col_mean = test_clean_df[col].mean()
                                            if abs(test_df_with_errors.loc[i, col] - col_mean) > 3 * col_std:
                                                error_type = 1  # 异常值
                                            else:
                                                error_type = 2  # 噪声

                                        error_locations.append((i, col, error_type))

                        st.session_state['test_df_with_errors'] = test_df_with_errors
                        st.session_state['test_clean_df'] = test_clean_df
                        st.session_state['test_error_locations'] = error_locations
                        st.success(f"Detected {len(error_locations)} errors in the test data.")
                    else:
                        st.error("Error: Clean data and error data must have the same dimensions.")

            # 如果已检测到错误，提供运行比较的按钮
            if 'test_error_locations' in st.session_state:
                if st.button("Run Comparison with Uploaded Data"):
                    df_with_errors = st.session_state['test_df_with_errors']
                    clean_df = st.session_state['test_clean_df']
                    error_locations = st.session_state['test_error_locations']


                    # 创建一个虚拟的injector对象用于兼容现有代码
                    class VirtualInjector:
                        def __init__(self, error_locs):
                            self.error_locations = error_locs


                    injector = VirtualInjector(error_locations)
                    st.info(f"Using uploaded test data with {len(error_locations)} detected errors.")
                    run_comparison = True

        elif data_source == "Generate example test data":
            st.subheader("Generate Example Test Data")

            # 获取当前任务类型
            task_type = st.session_state.get('task_type', 'classification')

            # 数据生成参数
            col1, col2 = st.columns(2)
            with col1:
                n_samples = st.slider("Number of samples:", min_value=100, max_value=5000, value=500, step=100)
                n_features = st.slider("Number of features:", min_value=2, max_value=20, value=5, step=1)

            with col2:
                error_rate = st.slider("Error rate:", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    missing_ratio = st.slider("Missing values ratio:", min_value=0.0, max_value=1.0, value=0.33,
                                              step=0.01)
                with col_b:
                    outlier_ratio = st.slider("Outlier values ratio:", min_value=0.0, max_value=1.0, value=0.33,
                                              step=0.01)
                with col_c:
                    noise_ratio = st.slider("Noise ratio:", min_value=0.0, max_value=1.0, value=0.34, step=0.01)

                # 归一化比例总和为1
                total = missing_ratio + outlier_ratio + noise_ratio
                missing_ratio = missing_ratio / total
                outlier_ratio = outlier_ratio / total
                noise_ratio = noise_ratio / total

            st.info(
                f"Normalized ratios: Missing={missing_ratio:.2f}, Outliers={outlier_ratio:.2f}, Noise={noise_ratio:.2f}")

            # 生成示例数据的按钮
            if st.button("Generate Example Data"):
                with st.spinner("Generating example data..."):
                    # 生成干净数据
                    clean_df = generate_clean_data(
                        task_type=task_type,
                        n_samples=n_samples,
                        n_features=n_features
                    )

                    # 创建错误注入器
                    injector = ErrorInjector(clean_df)

                    # 注入错误
                    missing_err_rate = error_rate * missing_ratio
                    outlier_err_rate = error_rate * outlier_ratio
                    noise_err_rate = error_rate * noise_ratio

                    df_with_errors = injector.inject_missing_values(error_rate=missing_err_rate)
                    df_with_errors = injector.inject_outliers(error_rate=outlier_err_rate)
                    df_with_errors = injector.inject_noise(error_rate=noise_err_rate)

                    error_locations = injector.error_locations

                    # 保存到session state
                    st.session_state['example_clean_df'] = clean_df
                    st.session_state['example_df_with_errors'] = df_with_errors
                    st.session_state['example_injector'] = injector

                    st.success(f"Successfully generated example data with {len(error_locations)} errors.")

                    # 显示数据预览
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Clean Data Preview:")
                        st.dataframe(clean_df.head(5))
                    with col2:
                        st.write("Data with Errors Preview:")
                        st.dataframe(df_with_errors.head(5))

                    # 可视化生成的数据
                    if st.checkbox("Show Data Visualization", value=True):
                        # 显示数据预览
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Clean Data:")
                            fig = visualize_data(clean_df, task_type,
                                                 f"Generated Clean Data ({error_rate:.2f})", return_fig=True)
                            st.pyplot(fig)
                        with col2:
                            st.write("Data with Errors:")
                            fig = visualize_data(df_with_errors, task_type,
                                                 f"Generated Data with Errors ({error_rate:.2f})", return_fig=True)
                            st.pyplot(fig)

            # 如果已生成示例数据，提供运行比较的按钮
            if 'example_df_with_errors' in st.session_state:
                if st.button("Run Comparison with Generated Data"):
                    df_with_errors = st.session_state['example_df_with_errors']
                    clean_df = st.session_state['example_clean_df']
                    injector = st.session_state['example_injector']
                    error_locations = injector.error_locations
                    run_comparison = True

        # 只有当用户点击了相应按钮后才运行策略比较
        if run_comparison and df_with_errors is not None and clean_df is not None and error_locations is not None:
            with st.spinner("Computing strategy comparisons..."):
                # 从session_state获取环境和代理
                env = st.session_state['env']
                agent = st.session_state['agent']
                task_type = st.session_state.get('task_type', 'classification')

                # 如果使用新数据，更新环境
                if data_source != "Use data from previous steps":
                    # 为新数据创建环境
                    if task_type == 'classification':
                        ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
                    else:
                        ml_model = RandomForestRegressor(n_estimators=10, random_state=42)

                    model_type = st.session_state.get('model_type', 'random_forest')
                    env = DataCleaningEnv(df_with_errors, ml_model, error_locations,
                                          task_type=task_type, model_type=model_type)

                # 设置评估器
                X = clean_df.drop('target', axis=1)
                y = clean_df['target']
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
                model_type = st.session_state.get('model_type', 'random_forest')
                evaluator = ModelEvaluator(X_train, y_train, X_val, y_val,
                                           task_type=task_type, model_type=model_type)

                # 评估策略
                strat_perf = evaluate_strategies(task_type, df_with_errors, error_locations,
                                                 evaluator, env, agent)

                # 将策略性能存储到session_state
                st.session_state['strategy_performance'] = strat_perf

                # 收集动作统计
                env.df = df_with_errors.copy()
                env.current_error_idx = 0
                actions = []
                feature_importances = []
                error_types = []

                state = env.reset()
                for _ in range(len(error_locations)):
                    action = agent.act(state)
                    actions.append(action)
                    etype, f_imp, _, _, _ = state
                    feature_importances.append(f_imp)
                    error_types.append(etype)
                    next_state, _, done, _ = env.step(action)
                    state = next_state
                    if done:
                        break
                col1, col2 = st.columns([1, 2])
                with col1:
                    # 生成并显示策略对比图
                    st.subheader("Strategy Performance Comparison")

                    # 创建性能比较图
                    fig, ax = plt.subplots(figsize=(5, 6))

                    strategies = ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']

                    if task_type == 'classification':
                        metrics = [strat_perf['do_nothing']['accuracy'],
                                   strat_perf['delete_all']['accuracy'],
                                   strat_perf['repair_all']['accuracy'],
                                   strat_perf['RL_optimal']['accuracy']]
                        metric_name = "Accuracy"
                    else:
                        metrics = [strat_perf['do_nothing']['r2_score'],
                                   strat_perf['delete_all']['r2_score'],
                                   strat_perf['repair_all']['r2_score'],
                                   strat_perf['RL_optimal']['r2_score']]
                        metric_name = "R² Score"

                    # 创建条形图
                    bars = ax.bar(strategies, metrics, color=['gray', 'red', 'blue', 'green'])

                    # 添加数据标签
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom')

                    ax.set_ylabel(metric_name)
                    ax.set_title('Strategy Performance Comparison')
                    ax.set_ylim(0, max(metrics) * 1.1)

                    plt.tight_layout()
                    st.pyplot(fig)

                # 如果是新生成或上传的数据，保存图像
                if data_source != "Use data from previous steps":
                    source_label = "custom" if data_source == "Upload new test data" else "generated"
                    fig_filename = f"strategy_comparison_{task_type}_{source_label}_data.png"
                    fig.savefig(fig_filename)
                    st.success(f"Saved strategy comparison as {fig_filename}")

                # 计算动作统计
                actions = np.array(actions)
                feature_importances = np.array(feature_importances)
                error_types = np.array(error_types)

                action_counts = {
                    'no_action': np.sum(actions == 0),
                    'repair': np.sum(actions == 1),
                    'delete': np.sum(actions == 2)
                }

                action_rates = {
                    'no_action': np.sum(actions == 0) / len(actions),
                    'repair': np.sum(actions == 1) / len(actions),
                    'delete': np.sum(actions == 2) / len(actions)
                }

                high_importance_mask = feature_importances > 0.5
                low_importance_mask = feature_importances <= 0.5
                missing_mask = np.array(error_types) == 0
                outlier_mask = np.array(error_types) == 1

                high_importance_actions = {
                    'no_action': np.sum(actions[high_importance_mask] == 0) / max(1, np.sum(high_importance_mask)),
                    'repair': np.sum(actions[high_importance_mask] == 1) / max(1, np.sum(high_importance_mask)),
                    'delete': np.sum(actions[high_importance_mask] == 2) / max(1, np.sum(high_importance_mask))
                }

                low_importance_actions = {
                    'no_action': np.sum(actions[low_importance_mask] == 0) / max(1, np.sum(low_importance_mask)),
                    'repair': np.sum(actions[low_importance_mask] == 1) / max(1, np.sum(low_importance_mask)),
                    'delete': np.sum(actions[low_importance_mask] == 2) / max(1, np.sum(low_importance_mask))
                }

                missing_actions = {
                    'no_action': np.sum(actions[missing_mask] == 0) / max(1, np.sum(missing_mask)),
                    'repair': np.sum(actions[missing_mask] == 1) / max(1, np.sum(missing_mask)),
                    'delete': np.sum(actions[missing_mask] == 2) / max(1, np.sum(missing_mask))
                }

                outlier_actions = {
                    'no_action': np.sum(actions[outlier_mask] == 0) / max(1, np.sum(outlier_mask)),
                    'repair': np.sum(actions[outlier_mask] == 1) / max(1, np.sum(outlier_mask)),
                    'delete': np.sum(actions[outlier_mask] == 2) / max(1, np.sum(outlier_mask))
                }

                # 将动作统计存储到session_state
                st.session_state['action_stats'] = {
                    'action_counts': action_counts,
                    'action_rates': action_rates,
                    'high_importance_actions': high_importance_actions,
                    'low_importance_actions': low_importance_actions,
                    'missing_actions': missing_actions,
                    'outlier_actions': outlier_actions
                }

                st.session_state['step3_complete'] = True

                # 显示动作分布
                st.subheader("Action Distribution Statistics")

                # 综合动作分布图 - 紧凑版
                fig, axs = plt.subplots(1, 4, figsize=(12, 4))
                action_stats = st.session_state['action_stats']
                # 1. 整体动作计数
                action_names = ['No Action', 'Repair', 'Delete']
                action_values = [action_stats['action_counts']['no_action'],
                                 action_stats['action_counts']['repair'],
                                 action_stats['action_counts']['delete']]

                bars = axs[0].bar(action_names, action_values, width=0.5, color=['gray', 'blue', 'red'])
                axs[0].set_xlabel("Action Type")
                axs[0].set_ylabel("Count")
                axs[0].set_title("Overall Action Distribution")

                for i, count in enumerate(action_values):
                    axs[0].text(i, count + 0.5, str(count), ha='center')

                # 2. 按特征重要性 - 高重要性
                high_imp_values = [action_stats['high_importance_actions']['no_action'],
                                   action_stats['high_importance_actions']['repair'],
                                   action_stats['high_importance_actions']['delete']]

                bars = axs[1].bar(action_names, high_imp_values, width=0.5, color=['gray', 'blue', 'red'])
                axs[1].set_xlabel("Action Type")
                axs[1].set_ylabel("Proportion")
                axs[1].set_title("Actions for High Importance Features")
                axs[1].set_ylim(0, 1)

                for i, val in enumerate(high_imp_values):
                    axs[1].text(i, val + 0.02, f"{val:.2f}", ha='center')

                # 3. 按错误类型 - 缺失值
                missing_values = [action_stats['missing_actions']['no_action'],
                                  action_stats['missing_actions']['repair'],
                                  action_stats['missing_actions']['delete']]

                bars = axs[2].bar(action_names, missing_values, width=0.5, color=['gray', 'blue', 'red'])
                axs[2].set_xlabel("Action Type")
                axs[2].set_ylabel("Proportion")
                axs[2].set_title("Actions for Missing Values")
                axs[2].set_ylim(0, 1)

                for i, val in enumerate(missing_values):
                    axs[2].text(i, val + 0.02, f"{val:.2f}", ha='center')

                # 4. 按错误类型 - 异常值
                outlier_values = [action_stats['outlier_actions']['no_action'],
                                  action_stats['outlier_actions']['repair'],
                                  action_stats['outlier_actions']['delete']]

                bars = axs[3].bar(action_names, outlier_values, width=0.3, color=['gray', 'blue', 'red'])
                axs[3].set_xlabel("Action Type")
                axs[3].set_ylabel("Proportion")
                axs[3].set_title("Actions for Outliers/Noise")
                axs[3].set_ylim(0, 1)

                for i, val in enumerate(outlier_values):
                    axs[3].text(i, val + 0.02, f"{val:.2f}", ha='center')

                plt.tight_layout()
                st.pyplot(fig)

                # 如果是新生成或上传的数据，保存图像
                if data_source != "Use data from previous steps":
                    source_label = "custom" if data_source == "Upload new test data" else "generated"
                    fig_filename = f"action_distribution_{task_type}_{source_label}_data.png"
                    fig.savefig(fig_filename)
                    st.success(f"Saved action distribution as {fig_filename}")
                with col2:
                    # 特性重要性比较图 - 附加图，更直观显示差异
                    st.subheader("Action Strategy by Feature Importance & Error Type")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(3)
                    width = 0.2

                    ax.bar(x - width * 1.5, high_imp_values, width, label='High Importance', color='darkblue')
                    ax.bar(x - width * 0.5, low_importance_actions.values(), width, label='Low Importance',
                           color='skyblue')
                    ax.bar(x + width * 0.5, missing_values, width, label='Missing Values', color='orange')
                    ax.bar(x + width * 1.5, outlier_values, width, label='Outliers', color='red')

                    ax.set_xticks(x)
                    ax.set_xticklabels(action_names)
                    ax.set_ylabel('Proportion')
                    ax.set_title('Action Strategy Comparison')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)

                    plt.tight_layout()
                    st.pyplot(fig)

                if data_source != "Use data from previous steps":
                    source_label = "custom" if data_source == "Upload new test data" else "generated"
                    fig_filename = f"action_strategy_{task_type}_{source_label}_data.png"
                    fig.savefig(fig_filename)
                    st.success(f"Saved action strategy as {fig_filename}")

                # 显示结果摘要
                st.subheader(f"Results Summary - {data_source.replace('Use ', '').title()}")

                # 创建性能摘要表
                if task_type == 'classification':
                    perf_df = pd.DataFrame({
                        'Strategy': ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal'],
                        'Accuracy': [strat_perf['do_nothing']['accuracy'],
                                     strat_perf['delete_all']['accuracy'],
                                     strat_perf['repair_all']['accuracy'],
                                     strat_perf['RL_optimal']['accuracy']],
                        'Precision': [strat_perf['do_nothing'].get('precision', np.nan),
                                      strat_perf['delete_all'].get('precision', np.nan),
                                      strat_perf['repair_all'].get('precision', np.nan),
                                      strat_perf['RL_optimal'].get('precision', np.nan)],
                        'Recall': [strat_perf['do_nothing'].get('recall', np.nan),
                                   strat_perf['delete_all'].get('recall', np.nan),
                                   strat_perf['repair_all'].get('recall', np.nan),
                                   strat_perf['RL_optimal'].get('recall', np.nan)],
                        'F1 Score': [strat_perf['do_nothing'].get('f1_score', np.nan),
                                     strat_perf['delete_all'].get('f1_score', np.nan),
                                     strat_perf['repair_all'].get('f1_score', np.nan),
                                     strat_perf['RL_optimal'].get('f1_score', np.nan)]
                    })
                else:
                    perf_df = pd.DataFrame({
                        'Strategy': ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal'],
                        'R² Score': [strat_perf['do_nothing']['r2_score'],
                                     strat_perf['delete_all']['r2_score'],
                                     strat_perf['repair_all']['r2_score'],
                                     strat_perf['RL_optimal']['r2_score']],
                        'MSE': [strat_perf['do_nothing'].get('mse', np.nan),
                                strat_perf['delete_all'].get('mse', np.nan),
                                strat_perf['repair_all'].get('mse', np.nan),
                                strat_perf['RL_optimal'].get('mse', np.nan)],
                        'MAE': [strat_perf['do_nothing'].get('mae', np.nan),
                                strat_perf['delete_all'].get('mae', np.nan),
                                strat_perf['repair_all'].get('mae', np.nan),
                                strat_perf['RL_optimal'].get('mae', np.nan)]
                    })

                # 格式化数值
                for col in perf_df.columns:
                    if col != 'Strategy':
                        perf_df[col] = perf_df[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A")

                st.write("Performance Metrics:")
                st.table(perf_df)

                # 创建动作分布摘要
                action_df = pd.DataFrame({
                    'Action Type': ['No Action', 'Repair', 'Delete'],
                    'Count': [action_counts['no_action'], action_counts['repair'], action_counts['delete']],
                    'Percentage': [f"{action_rates['no_action'] * 100:.1f}%",
                                   f"{action_rates['repair'] * 100:.1f}%",
                                   f"{action_rates['delete'] * 100:.1f}%"]
                })

                st.write("Action Distribution:")
                st.table(action_df)

                # 添加按钮跳转到Step 4
                if st.button("Proceed to Model Comparison"):
                    st.rerun()

        # 如果步骤完成但未选择运行，提醒用户选择动作
        elif st.session_state.get('step2_complete', False) and not run_comparison:
            if data_source == "Use data from previous steps":
                st.info("Click 'Run Comparison with Existing Data' to continue.")
            elif data_source == "Upload new test data":
                if 'test_error_locations' not in st.session_state:
                    st.info("Please upload test data and clean data, then detect error locations.")
                else:
                    st.info("Click 'Run Comparison with Uploaded Data' to continue.")
            elif data_source == "Generate example test data":
                if 'example_df_with_errors' not in st.session_state:
                    st.info("Click 'Generate Example Data' to create test data.")
                else:
                    st.info("Click 'Run Comparison with Generated Data' to continue.")

elif page == "Step 4: Model Comparison":
    st.header("Step 4: Multi-Model Comparison & Tolerance Analysis")

    if not st.session_state.get('step3_complete', False):
        st.warning("Please complete Step 3 first.")
        if st.button("Go to Step 3"):
            st.rerun()
    else:
        # 模型选择部分
        st.subheader("Select Models for Comparison")
        task_type = st.session_state.get('task_type', 'classification')
        error_rate = st.session_state['error_rate']

        if task_type == 'classification':
            available_models = ["random_forest", "svm", "logistic_regression"]
        else:
            available_models = ["random_forest", "svm", "linear_regression"]

        selected_models = st.multiselect(
            "Select models to compare:",
            available_models,
            default=[st.session_state.get('model_type', available_models[0])]
        )

        # 选择是使用现有策略还是重新训练
        use_policy = st.radio(
            "Use existing RL policy or retrain for each model?",
            ["Use existing policy", "Retrain RL for each model"]
        )

        if st.button("Run Model Comparison"):
            if not selected_models:
                st.error("Please select at least one model.")
            else:
                with st.spinner("Running model comparison..."):
                    # 从session_state获取数据
                    clean_df = st.session_state['clean_df']
                    df_with_errors = st.session_state['df_with_errors']
                    injector = st.session_state['injector']  # 这里要确保injector在session_state中
                    agent = st.session_state['agent']

                    # 准备评估数据
                    X = clean_df.drop('target', axis=1)
                    y = clean_df['target']
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

                    # 初始化结果存储
                    model_results = {}

                    for model_type in selected_models:
                        st.text(f"Processing model: {model_type}")

                        # 检查是否已有训练好的模型
                        model_exists = check_model_exists(model_type, task_type, error_rate)

                        # 为该模型初始化评估器
                        evaluator = ModelEvaluator(X_train, y_train, X_val, y_val,
                                                   task_type=task_type, model_type=model_type)

                        if use_policy == "Use existing policy" or model_exists:
                            # 使用已训练的代理（或加载现有模型）
                            env = st.session_state['env']
                            env.model_type = model_type  # 更新环境中的模型类型

                            if model_exists and model_type != st.session_state.get('model_type'):
                                # 如果存在针对当前模型的预训练模型，加载它
                                current_agent = load_model(model_type, task_type, error_rate)
                                if current_agent:
                                    agent = current_agent
                        else:
                            # 为该模型类型创建新环境
                            if task_type == 'classification':
                                if model_type == 'svm':
                                    ml_model = SVC(probability=True, random_state=42)
                                elif model_type == 'logistic_regression':
                                    ml_model = LogisticRegression(random_state=42, max_iter=1000)
                                else:
                                    ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
                            else:
                                if model_type == 'svm':
                                    ml_model = SVR()
                                elif model_type == 'linear_regression':
                                    ml_model = LinearRegression()
                                else:
                                    ml_model = RandomForestRegressor(n_estimators=10, random_state=42)

                            # 创建新环境
                            env = DataCleaningEnv(df_with_errors.copy(), ml_model, injector.error_locations,
                                                  task_type=task_type, model_type=model_type)

                            # 训练新代理
                            new_agent = DQNAgent(state_size=5, action_size=3)

                            # 快速训练（较少轮次）
                            n_quick_episodes = 20
                            for e in range(n_quick_episodes):
                                state = env.reset()
                                for _ in range(len(injector.error_locations)):
                                    action = new_agent.act(state)
                                    next_state, reward, done, _ = env.step(action)
                                    new_agent.remember(state, action, reward, next_state, done)
                                    state = next_state
                                    if done:
                                        break
                                new_agent.replay(batch_size=32)

                            # 使用新训练的代理
                            agent = new_agent

                            # 保存训练好的模型
                            save_model(agent, model_type, task_type, error_rate)

                        # 用当前模型评估策略
                        strat_perf = evaluate_strategies(task_type, df_with_errors,
                                                         injector.error_locations, evaluator, env, agent)

                        # 收集动作统计
                        env.df = df_with_errors.copy()
                        env.current_error_idx = 0
                        actions = []

                        state = env.reset()
                        for _ in range(len(injector.error_locations)):
                            action = agent.act(state)
                            actions.append(action)
                            next_state, _, done, _ = env.step(action)
                            state = next_state
                            if done:
                                break

                        actions = np.array(actions)
                        action_counts = {
                            'no_action': np.sum(actions == 0),
                            'repair': np.sum(actions == 1),
                            'delete': np.sum(actions == 2)
                        }

                        # 存储该模型的结果
                        model_results[model_type] = {
                            'strategy_performance': strat_perf,
                            'action_counts': action_counts
                        }

                    # 将模型比较结果存储到session_state
                    st.session_state['model_comparison'] = model_results
                    st.session_state['step4_complete'] = True

                st.success("Model comparison complete!")

        # 如果有模型比较结果，显示结果
        if st.session_state.get('model_comparison'):
            st.subheader("Model Comparison Results")

            model_results = st.session_state['model_comparison']

            # 性能指标比较
            st.write("Performance Metrics Across Models:")

            # 优化Step 4的模型比较图表
            if task_type == 'classification':
                # 计算y轴的合理范围
                all_accs = []
                for model_name in model_results.keys():
                    strat_perf = model_results[model_name]['strategy_performance']
                    all_accs.extend([
                        strat_perf['do_nothing']["accuracy"],
                        strat_perf['delete_all']["accuracy"],
                        strat_perf['repair_all']["accuracy"],
                        strat_perf['RL_optimal']["accuracy"]
                    ])

                min_acc = max(0.5, min(all_accs) - 0.05)  # 从0.5开始以突出差异
                max_acc = min(1, max(all_accs) + 0.05)

                # 创建一个紧凑的准确率比较图
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    bar_width = 0.15
                    index = np.arange(4)

                    for i, model_name in enumerate(model_results.keys()):
                        strat_perf = model_results[model_name]['strategy_performance']
                        accuracies = [strat_perf['do_nothing']["accuracy"],
                                      strat_perf['delete_all']["accuracy"],
                                      strat_perf['repair_all']["accuracy"],
                                      strat_perf['RL_optimal']["accuracy"]]

                        bars = ax.bar(index + i * bar_width, accuracies, bar_width, label=model_name)

                        for j, bar in enumerate(bars):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                                    f'{accuracies[j]:.3f}', ha='center', va='bottom',
                                    rotation=45, fontsize=8)

                    ax.set_xlabel('Strategy')
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Model Accuracy by Strategy')
                    ax.set_xticks(index + bar_width * (len(model_results) - 1) / 2)
                    ax.set_xticklabels(['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal'], rotation=20)
                    ax.legend(loc='lower right', fontsize=8)
                    ax.set_ylim(min_acc, max_acc)
                    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    # 折线图比较
                    fig, ax = plt.subplots(figsize=(6, 4))
                    strategies = ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']

                    for model_name in model_results.keys():
                        strat_perf = model_results[model_name]['strategy_performance']
                        accuracies = [strat_perf['do_nothing']["accuracy"],
                                      strat_perf['delete_all']["accuracy"],
                                      strat_perf['repair_all']["accuracy"],
                                      strat_perf['RL_optimal']["accuracy"]]

                        ax.plot(strategies, accuracies, 'o-', linewidth=2, markersize=6, label=model_name)

                        for i, acc in enumerate(accuracies):
                            ax.text(i, acc + 0.005, f'{acc:.3f}', ha='center', fontsize=8)

                    ax.set_xlabel('Strategy')
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Accuracy Trend Across Strategies')
                    ax.legend(fontsize=8)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_ylim(min_acc, max_acc)
                    plt.tight_layout()
                    st.pyplot(fig)

            # 动作分布比较
            # 修改动作分布比较图 - 更紧凑的布局
            # 创建一个组合图表来显示动作分布和策略比较
            st.subheader("Action Distribution & Model Strategy Analysis")

            # 优化动作分布比较
            col1, col2 = st.columns([3, 2])

            with col1:
                # 左侧 - 条形图比较各模型的动作分布
                fig, ax = plt.subplots(figsize=(8, 5))
                bar_width = 0.15
                index = np.arange(3)  # 3种动作类型

                for i, model_name in enumerate(model_results.keys()):
                    action_counts = model_results[model_name]['action_counts']
                    total = sum(action_counts.values())
                    counts_pct = [action_counts['no_action'] / total,
                                  action_counts['repair'] / total,
                                  action_counts['delete'] / total]

                    ax.bar(index + i * bar_width, counts_pct, bar_width, label=model_name)

                    for j, pct in enumerate(counts_pct):
                        ax.text(index[j] + i * bar_width, pct + 0.02, f"{pct:.0%}",
                                ha='center', va='bottom', fontsize=8, rotation=45)

                ax.set_xlabel('Action Type')
                ax.set_ylabel('Proportion')
                ax.set_title('Action Distribution by Model')
                ax.set_xticks(index + bar_width * (len(model_results) - 1) / 2)
                ax.set_xticklabels(['No Action', 'Repair', 'Delete'])
                ax.legend()
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                # 右侧 - 饼图显示整体的动作偏好
                fig, ax = plt.subplots(figsize=(5, 5))
                total_actions = {'no_action': 0, 'repair': 0, 'delete': 0}
                for model_name in model_results.keys():
                    for action_type in total_actions:
                        total_actions[action_type] += model_results[model_name]['action_counts'][action_type]

                total_sum = sum(total_actions.values())
                action_labels = ['No Action', 'Repair', 'Delete']
                sizes = [total_actions['no_action'] / total_sum,
                         total_actions['repair'] / total_sum,
                         total_actions['delete'] / total_sum]
                colors = ['#4878d0', '#ee854a', '#d64646']
                explode = (0.05, 0, 0)  # 突出"No Action"

                wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=action_labels,
                                                  colors=colors, autopct='%1.1f%%',
                                                  shadow=True, startangle=90,
                                                  textprops={'fontsize': 9})
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                ax.axis('equal')
                ax.set_title('Overall Action Distribution')
                plt.tight_layout()
                st.pyplot(fig)

            # 容忍度阈值分析部分
            # 改进容忍度阈值分析 - 使用可视化代替长篇文字说明
            st.subheader("Data Quality Tolerance Analysis")

            # 使用热图和雷达图来可视化容忍度阈值
            tolerance_data = {
                'overall_tolerance': st.session_state['error_rate'] * 0.8,
                'repair_vs_delete': st.session_state['error_rate'] * 0.6,
                'key_feature_tolerance': st.session_state['error_rate'] * 0.7,
                'non_key_feature_tolerance': st.session_state['error_rate'] * 0.9,
                'missing_tolerance': st.session_state['error_rate'] * 0.75,
                'outlier_tolerance': st.session_state['error_rate'] * 0.65
            }

            # 创建一个热图显示不同阈值
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # 左侧：热图
            tolerance_df = pd.DataFrame({
                'Threshold Type': ['Overall', 'Repair vs Delete', 'Key Features', 'Non-key Features', 'Missing Values',
                                   'Outliers'],
                'Tolerance Value': [tolerance_data['overall_tolerance'],
                                    tolerance_data['repair_vs_delete'],
                                    tolerance_data['key_feature_tolerance'],
                                    tolerance_data['non_key_feature_tolerance'],
                                    tolerance_data['missing_tolerance'],
                                    tolerance_data['outlier_tolerance']]
            })

            # 将数据转为热图格式
            heatmap_data = tolerance_df.set_index('Threshold Type')
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1)
            ax1.set_title('Data Quality Tolerance Thresholds')
            ax1.set_ylabel('')

            # 右侧：容忍度比较图
            categories = ['Overall', 'Repair/Delete', 'Key Features',
                          'Non-key Features', 'Missing', 'Outliers']
            values = [tolerance_data['overall_tolerance'],
                      tolerance_data['repair_vs_delete'],
                      tolerance_data['key_feature_tolerance'],
                      tolerance_data['non_key_feature_tolerance'],
                      tolerance_data['missing_tolerance'],
                      tolerance_data['outlier_tolerance']]

            # 关闭右侧的轴线
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)

            # 创建条形图
            bars = ax2.barh(categories, values, color=plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(categories))))
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Tolerance Threshold')
            ax2.set_title('Tolerance Threshold Analysis')
            ax2.grid(True, linestyle='--', alpha=0.3)

            # 添加数值标签
            for i, v in enumerate(values):
                ax2.text(v + 0.02, i, f'{v:.3f}', va='center')

            plt.tight_layout()
            st.pyplot(fig)

            # 改进模型性能vs错误率的可视化
            fig, ax = plt.subplots(figsize=(10, 6))

            # 不同错误率下的性能
            error_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            # 获取当前数据范围
            all_perf_values = []

            for model_name in model_results.keys():
                current_perf = model_results[model_name]['strategy_performance']['RL_optimal']

                if task_type == 'classification':
                    base_perf = current_perf["accuracy"]
                else:
                    base_perf = current_perf["r2_score"]

                simulated_perf = [
                    min(1.0, base_perf * (1 + 0.1 * (1 - er))) if er < st.session_state['error_rate']
                    else base_perf * (1 - 0.2 * (er - st.session_state['error_rate']))
                    for er in error_rates
                ]
                all_perf_values.extend(simulated_perf)

            # 计算合理的y轴范围
            min_perf = max(0, min(all_perf_values) - 0.05)
            max_perf = min(1, max(all_perf_values) + 0.05)

            for model_name in model_results.keys():
                current_perf = model_results[model_name]['strategy_performance']['RL_optimal']

                if task_type == 'classification':
                    base_perf = current_perf["accuracy"]
                    metric_name = "Accuracy"
                else:
                    base_perf = current_perf["r2_score"]
                    metric_name = "R² Score"

                simulated_perf = [
                    min(1.0, base_perf * (1 + 0.1 * (1 - er))) if er < st.session_state['error_rate']
                    else base_perf * (1 - 0.2 * (er - st.session_state['error_rate']))
                    for er in error_rates
                ]

                ax.plot(error_rates, simulated_perf, 'o-', linewidth=2, markersize=8, label=model_name)

                # 用横跨的线标识容忍度阈值
                threshold_idx = next(
                    (i for i, er in enumerate(error_rates) if er > tolerance_data['overall_tolerance']), None)
                if threshold_idx is not None and threshold_idx > 0:
                    # 插值找到阈值对应的性能点
                    perf_at_threshold = np.interp(tolerance_data['overall_tolerance'],
                                                  [error_rates[threshold_idx - 1], error_rates[threshold_idx]],
                                                  [simulated_perf[threshold_idx - 1], simulated_perf[threshold_idx]])

                    ax.plot([tolerance_data['overall_tolerance'], tolerance_data['overall_tolerance']],
                            [min_perf, perf_at_threshold], ':', color='gray', alpha=0.5)

                    ax.plot([0, tolerance_data['overall_tolerance']],
                            [perf_at_threshold, perf_at_threshold], ':', color='gray', alpha=0.5)

                    ax.text(tolerance_data['overall_tolerance'] - 0.02, min_perf + 0.02,
                            f'Threshold\n{tolerance_data["overall_tolerance"]:.2f}',
                            ha='right', va='bottom', fontsize=8)

            # 标记当前错误率
            ax.axvline(x=st.session_state['error_rate'], color='r', linestyle='--',
                       label=f'Current Error Rate: {st.session_state["error_rate"]:.2f}')

            # 添加性能阈值线
            ax.axhline(y=0.7, color='k', linestyle=':', label='Performance Threshold (0.7)')

            ax.set_xlabel('Error Rate')
            ax.set_ylabel(metric_name)
            ax.set_title('Model Performance vs Error Rate with Tolerance Threshold')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.7)

            # 调整y轴范围以突出差异
            ax.set_ylim(min_perf, max_perf)

            plt.tight_layout()
            st.pyplot(fig)

            # 添加按钮跳转到Step 5
            if st.button("Proceed to Export Results"):
                st.rerun()

elif page == "Step 5: Export Results":
    st.header("Step 5: Export Results & Analysis Report")

    if not st.session_state.get('step4_complete', False):
        st.warning("Please complete Step 4 first.")
        if st.button("Go to Step 4"):
            st.rerun()
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;text-align:center;">
                <h4 style="margin:0;color:#666;">Task Type</h4>
                <h2 style="margin:5px 0;color:#1f77b4;">{st.session_state.get('task_type', 'classification').upper()}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;text-align:center;">
                <h4 style="margin:0;color:#666;">Total Error Rate</h4>
                <h2 style="margin:5px 0;color:#1f77b4;">{st.session_state['error_rate']:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;text-align:center;">
                <h4 style="margin:0;color:#666;">Error Composition</h4>
                <div style="margin:5px 0;font-size:0.9em;">
                    <span style="color:#e74c3c;font-weight:bold;">Missing: {st.session_state['missing_ratio']:.2f}</span> | 
                    <span style="color:#3498db;font-weight:bold;">Outliers: {st.session_state['outlier_ratio']:.2f}</span> | 
                    <span style="color:#2ecc71;font-weight:bold;">Noise: {st.session_state['noise_ratio']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 汇总性能数据并可视化
        st.subheader("Performance Analysis")

        model_results = st.session_state['model_comparison']
        task_type = st.session_state.get('task_type', 'classification')

        # 创建摘要数据
        summary_data = []
        for model_name, results in model_results.items():
            strat_perf = results['strategy_performance']
            if task_type == 'classification':
                summary_data.append({
                    'Model': model_name,
                    'Do Nothing': strat_perf['do_nothing']['accuracy'],
                    'Delete All': strat_perf['delete_all']['accuracy'],
                    'Repair All': strat_perf['repair_all']['accuracy'],
                    'RL Optimal': strat_perf['RL_optimal']['accuracy'],
                    'Metric': 'Accuracy'
                })
            else:
                summary_data.append({
                    'Model': model_name,
                    'Do Nothing': strat_perf['do_nothing']['r2_score'],
                    'Delete All': strat_perf['delete_all']['r2_score'],
                    'Repair All': strat_perf['repair_all']['r2_score'],
                    'RL Optimal': strat_perf['RL_optimal']['r2_score'],
                    'Metric': 'R² Score'
                })

        # 将数据转为DataFrame以便显示
        summary_df_numeric = pd.DataFrame(summary_data)

        # 创建格式化版本以便显示
        summary_df_display = summary_df_numeric.copy()
        for col in ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']:
            summary_df_display[col] = summary_df_display[col].apply(lambda x: f"{x:.3f}")

        # 并排显示性能摘要和可视化
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Strategy Performance")
            st.dataframe(summary_df_display.drop('Metric', axis=1), use_container_width=True)

            # 动作分布摘要 - 紧凑格式
            st.markdown("#### Action Distribution")

            action_summary = []
            for model_name, results in model_results.items():
                action_counts = results['action_counts']
                total = sum(action_counts.values())
                action_summary.append({
                    'Model': model_name,
                    'No Action': f"{action_counts['no_action'] / total * 100:.1f}%",
                    'Repair': f"{action_counts['repair'] / total * 100:.1f}%",
                    'Delete': f"{action_counts['delete'] / total * 100:.1f}%",
                })

            action_summary_df = pd.DataFrame(action_summary)
            st.dataframe(action_summary_df, use_container_width=True)

        with col2:
            # 创建性能可视化
            fig, ax = plt.subplots(figsize=(8, 4))

            # 计算Y轴范围，减小范围以强调差异
            if task_type == 'classification':
                all_values = summary_df_numeric[
                    ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']].values.flatten()
                metric_name = "Accuracy"
            else:
                all_values = summary_df_numeric[
                    ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']].values.flatten()
                metric_name = "R² Score"

            min_val = max(0, np.min(all_values) - 0.05)
            max_val = min(1, np.max(all_values) + 0.05)

            # 为每个模型绘制折线图，突出最优策略
            strategies = ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']
            markers = ['o', 's', '^', 'D']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            for i, model_name in enumerate(summary_df_numeric['Model']):
                values = summary_df_numeric.loc[i, ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']].values
                ax.plot(strategies, values, marker=markers[i % len(markers)], markersize=8,
                        linewidth=2, label=model_name, color=colors[i % len(colors)])

                # 标记 RL Optimal 的值
                ax.text(3, values[3], f'{values[3]:.3f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=9, color=colors[i % len(colors)])

            ax.set_xlabel('Strategy')
            ax.set_ylabel(metric_name)
            ax.set_ylim(min_val, max_val)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(title="Models", loc='upper left', fontsize=9)
            ax.set_title(f'Strategy Performance Comparison ({metric_name})')

            plt.tight_layout()
            st.pyplot(fig)

        # 容忍度阈值分析 - 使用可视化代替文本
        st.subheader("Tolerance Threshold Analysis")

        # 容忍度阈值设置
        tolerance_data = {
            'overall_tolerance': st.session_state['error_rate'] * 0.8,
            'repair_vs_delete': st.session_state['error_rate'] * 0.6,
            'key_feature_tolerance': st.session_state['error_rate'] * 0.7,
            'non_key_feature_tolerance': st.session_state['error_rate'] * 0.9,
            'missing_tolerance': st.session_state['error_rate'] * 0.75,
            'outlier_tolerance': st.session_state['error_rate'] * 0.65
        }

        # 使用三列布局展示关键阈值
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Data Quality Threshold", f"{tolerance_data['overall_tolerance']:.3f}",
                      f"{tolerance_data['overall_tolerance'] * 100:.1f}% of error rate")

        with col2:
            st.metric("Repair vs Delete", f"{tolerance_data['repair_vs_delete']:.3f}",
                      "Switching point")

        with col3:
            # 确定哪种类型的错误更重要
            error_type = "Outliers" if tolerance_data['missing_tolerance'] > tolerance_data[
                'outlier_tolerance'] else "Missing Values"
            diff = abs(tolerance_data['missing_tolerance'] - tolerance_data['outlier_tolerance'])
            st.metric("Priority Focus", error_type, f"{diff:.3f} difference")

        # 视觉化容忍度比较
        col1, col2 = st.columns([5, 2])

        with col1:
            # 创建容忍度路线图
            fig, ax = plt.subplots(figsize=(8, 3))

            # 绘制容忍度区域和策略
            ax.axvspan(0, tolerance_data['overall_tolerance'], alpha=0.2, color='#2ecc71', label='Direct Use')
            ax.axvspan(tolerance_data['overall_tolerance'], tolerance_data['repair_vs_delete'], alpha=0.2,
                       color='#f1c40f', label='Quality Check')
            ax.axvspan(tolerance_data['repair_vs_delete'], 1, alpha=0.2, color='#e74c3c', label='Quality Critical')

            # 添加当前错误率标记
            ax.axvline(x=st.session_state['error_rate'], color='black', linestyle='-', linewidth=2,
                       label=f'Current Error Rate: {st.session_state["error_rate"]:.2f}')

            # 添加文本标签
            ax.text(tolerance_data['overall_tolerance'] / 2, 0.5, "Direct Use",
                    ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text((tolerance_data['overall_tolerance'] + tolerance_data['repair_vs_delete']) / 2, 0.5, "Repair",
                    ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text((tolerance_data['repair_vs_delete'] + 1) / 2, 0.5, "Delete",
                    ha='center', va='center', fontsize=10, fontweight='bold')

            # 设置图表属性
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Error Rate')
            ax.set_title('Error Rate Tolerance Strategy Map')

            # 添加阈值标记
            for threshold, name in [(tolerance_data['overall_tolerance'], 'Quality\nThreshold'),
                                    (tolerance_data['repair_vs_delete'], 'Repair/Delete\nThreshold')]:
                ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.7)
                ax.text(threshold, 0.1, f"{threshold:.3f}", ha='center', va='bottom',
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
                ax.text(threshold, 0.9, name, ha='center', va='top', fontsize=8,
                        rotation=90, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            # 雷达图
            categories = ['Overall', 'Repair/Delete', 'Key Features',
                          'Non-key Features', 'Missing Values', 'Outliers']
            values = [tolerance_data['overall_tolerance'],
                      tolerance_data['repair_vs_delete'],
                      tolerance_data['key_feature_tolerance'],
                      tolerance_data['non_key_feature_tolerance'],
                      tolerance_data['missing_tolerance'],
                      tolerance_data['outlier_tolerance']]

            # 闭合多边形
            values_for_radar = values + [values[0]]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]

            # 计算最大值用于设置更合适的Y轴上限（让图更“撑开”）
            max_value = max(values)
            ymax = min(1.0, max_value + 0.2)  # 自动向上扩展20%，最多不超过1

            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

            # 绘制区域
            ax.fill(angles, values_for_radar, color='#3498db', alpha=0.3)
            ax.plot(angles, values_for_radar, color='#3498db', linewidth=2)

            # 数据标签
            for i, (angle, value) in enumerate(zip(angles[:-1], values)):
                ax.text(angle, value + ymax * 0.05, f"{value:.2f}",
                        ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

            # 设置刻度和网格
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.set_ylim(0, ymax)
            ax.set_yticks(np.linspace(0, ymax, 5))  # 添加网格线
            ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(0, ymax, 5)], fontsize=8, color='gray')
            ax.yaxis.grid(True, linestyle='dashed', alpha=0.4)
            ax.xaxis.grid(True, linestyle='dotted', alpha=0.4)

            # 标题
            ax.set_title('Tolerance Thresholds by Category', pad=20, fontsize=12)

            plt.tight_layout()
            st.pyplot(fig)

        # 详细的阈值说明 - 放入可折叠区域
        with st.expander("View Detailed Tolerance Analysis", expanded=False):
            st.markdown("### Tolerance Thresholds Report")

            st.markdown("#### 1. Overall Tolerance Threshold")
            st.markdown(f"- **Overall data quality tolerance threshold:** {tolerance_data['overall_tolerance']:.3f}")
            st.markdown(
                f"- When error rate exceeds {tolerance_data['overall_tolerance'] * 100:.1f}%, data preprocessing is required.")

            st.markdown("#### 2. Repair vs. Delete Strategy Threshold")
            st.markdown(f"- **Repair/Delete threshold:** {tolerance_data['repair_vs_delete']:.3f}")
            st.markdown(
                f"- When error rate is below {tolerance_data['repair_vs_delete'] * 100:.1f}%, repair strategy is preferred.")
            st.markdown(
                f"- When error rate is above {tolerance_data['repair_vs_delete'] * 100:.1f}%, delete strategy is preferred.")

            st.markdown("#### 3. Feature Importance Analysis")
            st.markdown(f"- **Key feature tolerance threshold:** {tolerance_data['key_feature_tolerance']:.3f}")
            st.markdown(f"- **Non-key feature tolerance threshold:** {tolerance_data['non_key_feature_tolerance']:.3f}")
            diff = tolerance_data['non_key_feature_tolerance'] - tolerance_data['key_feature_tolerance']
            st.markdown(f"- **Conclusion:** Key features have lower tolerance (difference: {diff:.3f})")
            st.markdown("  - Models require higher data quality for important features.")

            st.markdown("#### 4. Error Type Analysis")
            st.markdown(f"- **Missing value tolerance threshold:** {tolerance_data['missing_tolerance']:.3f}")
            st.markdown(f"- **Outlier tolerance threshold:** {tolerance_data['outlier_tolerance']:.3f}")
            diff = tolerance_data['missing_tolerance'] - tolerance_data['outlier_tolerance']
            if diff > 0:
                st.markdown(f"- **Conclusion:** Models have lower tolerance for outliers (difference: {abs(diff):.3f})")
                st.markdown("  - Priority should be given to handling outliers.")
            else:
                st.markdown(
                    f"- **Conclusion:** Models have lower tolerance for missing values (difference: {abs(diff):.3f})")
                st.markdown("  - Priority should be given to handling missing values.")


        # 确定最佳模型
        def determine_best_model(model_results, task_type):
            """
            确定最佳模型，基于多个评估因素

            参数:
            - model_results: 包含各模型性能结果的字典
            - task_type: 'classification' 或 'regression'

            返回:
            - best_model: 最佳模型名称
            - model_scores: 各模型的详细评分
            """
            # 初始化评分记录
            model_scores = {}

            for model_name, results in model_results.items():
                strat_perf = results['strategy_performance']
                action_counts = results['action_counts']
                total_actions = sum(action_counts.values())

                # 1. 基本性能分数 (0-100分)
                if task_type == 'classification':
                    base_score = strat_perf['RL_optimal']["accuracy"] * 100
                    baseline_score = strat_perf['do_nothing']["accuracy"] * 100
                else:
                    base_score = strat_perf['RL_optimal']["r2_score"] * 100
                    baseline_score = strat_perf['do_nothing']["r2_score"] * 100

                # 2. 改进幅度 (0-50分)
                # 计算相对于不做任何操作的改进百分比
                improvement = base_score - baseline_score
                improvement_score = min(50, max(0, improvement * 5))  # 每提高1个百分点得5分，最高50分

                # 3. 鲁棒性评分 (0-30分)
                # 计算不同策略下的性能稳定性
                if task_type == 'classification':
                    performances = [
                        strat_perf['do_nothing']["accuracy"],
                        strat_perf['delete_all']["accuracy"],
                        strat_perf['repair_all']["accuracy"],
                        strat_perf['RL_optimal']["accuracy"]
                    ]
                else:
                    performances = [
                        strat_perf['do_nothing']["r2_score"],
                        strat_perf['delete_all']["r2_score"],
                        strat_perf['repair_all']["r2_score"],
                        strat_perf['RL_optimal']["r2_score"]
                    ]

                # 计算性能的标准差 (越低越稳定)
                stability = np.std(performances)
                stability_score = 30 * (1 - min(stability * 5, 1))  # 标准差越低，得分越高

                # 4. 策略效率 (0-20分)
                # 评估模型的动作选择是否经济高效
                # 假设修复比删除更高效，无操作表示不需要处理
                repair_rate = action_counts['repair'] / total_actions if total_actions > 0 else 0
                delete_rate = action_counts['delete'] / total_actions if total_actions > 0 else 0
                no_action_rate = action_counts['no_action'] / total_actions if total_actions > 0 else 0

                # 首选修复而非删除，但也要保持一定平衡
                strategy_score = 20 * (0.5 * repair_rate + 0.3 * no_action_rate + 0.2 * (1 - delete_rate))

                # 5. 综合评分 (0-200分)
                total_score = base_score + improvement_score + stability_score + strategy_score

                # 记录详细评分
                model_scores[model_name] = {
                    'base_score': base_score,
                    'improvement_score': improvement_score,
                    'stability_score': stability_score,
                    'strategy_score': strategy_score,
                    'total_score': total_score
                }

            # 选择综合评分最高的模型
            best_model = max(model_scores.items(), key=lambda x: x[1]['total_score'])[0]

            return best_model, model_scores


        # 确定最佳模型并显示评分结果
        best_model, model_scores = determine_best_model(model_results, task_type)

        # 最终建议部分 - 美观卡片式布局
        st.subheader("Final Recommendations")

        col1, col2 = st.columns(2)

        # 左侧显示最佳模型和策略
        with col1:
            st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:10px;">
                <h3 style="margin:0;color:#1f77b4;">Best Model</h3>
                <h2 style="margin:5px 0;color:#d62728;">{best_model.upper()}</h2>
                <p style="margin:0;font-size:0.9em;">Score: {model_scores[best_model]['total_score']:.1f}/200</p>
            </div>

            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;">
                <h3 style="margin:0;color:#1f77b4;">Best Strategy</h3>
                <h2 style="margin:5px 0;color:#d62728;">RL Optimal</h2>
                <p style="margin:0;font-size:0.9em;">Performance: {model_scores[best_model]['base_score']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # 右侧显示重要阈值
        with col2:
            st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:10px;">
                <h3 style="margin:0;color:#1f77b4;">Tolerance Threshold</h3>
                <h2 style="margin:5px 0;color:#d62728;">{tolerance_data['overall_tolerance']:.3f}</h2>
                <p style="margin:0;font-size:0.9em;">Critical point for data quality</p>
            </div>

            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;">
                <h3 style="margin:0;color:#1f77b4;">Error Type Focus</h3>
                <h2 style="margin:5px 0;color:#d62728;">
                {'Outliers' if tolerance_data['missing_tolerance'] > tolerance_data['outlier_tolerance'] else 'Missing Values'}
                </h2>
                <p style="margin:0;font-size:0.9em;">Priority for data cleaning</p>
            </div>
            """, unsafe_allow_html=True)

        # 模型评分可视化
        st.subheader("Model Score Breakdown")

        # 创建评分表格数据
        score_data = []
        for model_name, scores in model_scores.items():
            score_data.append({
                'Model': model_name,
                'Base Performance': f"{scores['base_score']:.1f}",
                'Improvement': f"{scores['improvement_score']:.1f}",
                'Stability': f"{scores['stability_score']:.1f}",
                'Strategy': f"{scores['strategy_score']:.1f}",
                'Total Score': f"{scores['total_score']:.1f}"
            })

        score_df = pd.DataFrame(score_data)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(score_df, use_container_width=True)

            with st.expander("Score Calculation Methodology", expanded=False):
                st.markdown("""
                **Model scoring is based on:**

                1. **Base Performance (0-100)**: Raw performance of RL optimal strategy
                2. **Improvement (0-50)**: Improvement over "do nothing" baseline
                3. **Stability (0-30)**: Consistency across different strategies
                4. **Strategy (0-20)**: Preference for efficient repair over deletion

                Total score is the sum of all components (max 200).
                """)

        with col2:
            # 创建堆叠条形图展示评分细分
            fig, ax = plt.subplots(figsize=(8, 4))

            models = list(model_scores.keys())
            x = np.arange(len(models))
            width = 0.6

            # 准备数据
            base_scores = [model_scores[m]['base_score'] for m in models]
            improvement_scores = [model_scores[m]['improvement_score'] for m in models]
            stability_scores = [model_scores[m]['stability_score'] for m in models]
            strategy_scores = [model_scores[m]['strategy_score'] for m in models]

            # 创建堆叠条形图
            p1 = ax.bar(x, base_scores, width, label='Base Performance', color='#3498db')
            p2 = ax.bar(x, improvement_scores, width, bottom=base_scores, label='Improvement', color='#2ecc71')

            bottom = np.array(base_scores) + np.array(improvement_scores)
            p3 = ax.bar(x, stability_scores, width, bottom=bottom, label='Stability', color='#f1c40f')

            bottom = bottom + np.array(stability_scores)
            p4 = ax.bar(x, strategy_scores, width, bottom=bottom, label='Cost', color='#e74c3c')

            # 添加总分标签
            for i, model in enumerate(models):
                total = model_scores[model]['total_score']
                ax.text(i, total + 5, f'{total:.1f}', ha='center', va='bottom', fontweight='bold')

                # 如果是最佳模型，添加标记
                if model == best_model:
                    ax.text(i, total + 15, '★ BEST', ha='center', va='bottom',
                            color='#e74c3c', fontweight='bold', fontsize=12)

            # 设置图表属性
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Score Breakdown')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_ylim(0, max([model_scores[m]['total_score'] for m in models]) * 1.2)

            plt.tight_layout()
            st.pyplot(fig)

        # 导出选项部分 - 使用卡片式布局
        st.subheader("Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;text-align:center;">
                <h4 style="margin:0;color:#666;">Data Export</h4>
                <p style="margin:10px 0;">Export processed data and results</p>
            </div>
            """, unsafe_allow_html=True)

            export_format = st.radio("Select export format:", ["CSV", "Excel", "JSON"])

            # 检查是否有已生成的图像文件
            task_type = st.session_state.get('task_type', 'classification')
            error_rate = st.session_state['error_rate']
            all_images = find_images(task_type, error_rate)

            if st.button("Export Data & Results"):
                with st.spinner("Preparing export..."):
                    # 获取RL处理后的清洁数据
                    env = st.session_state['env']
                    processed_df = env.df.copy()

                    # 创建带时间戳的文件名
                    timestamp = time.strftime("%Y%m%d-%H%M%S")

                    if export_format == "CSV":
                        # 导出为CSV
                        csv = processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download processed data (CSV)",
                            data=csv,
                            file_name=f"cbtad_processed_data_{timestamp}.csv",
                            mime="text/csv",
                        )

                        # 准备结果摘要
                        results_df = pd.DataFrame(summary_data)
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download results summary (CSV)",
                            data=csv_results,
                            file_name=f"cbtad_results_{timestamp}.csv",
                            mime="text/csv",
                        )
                    elif export_format == "Excel":
                        # 对于Excel导出，我们需要先保存到临时文件
                        excel_file = f"cbtad_export_{timestamp}.xlsx"
                        with pd.ExcelWriter(excel_file) as writer:
                            processed_df.to_excel(writer, sheet_name="Processed Data", index=False)
                            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Performance Summary", index=False)
                            pd.DataFrame(action_summary).to_excel(writer, sheet_name="Action Summary", index=False)

                            # 添加模型得分数据
                            score_df = pd.DataFrame(score_data)
                            score_df.to_excel(writer, sheet_name="Model Scores", index=False)

                            # 添加容忍度阈值数据
                            tolerance_df = pd.DataFrame([tolerance_data])
                            tolerance_df.to_excel(writer, sheet_name="Tolerance Thresholds", index=False)

                        # 读取文件并提供下载按钮
                        with open(excel_file, "rb") as f:
                            excel_data = f.read()

                        st.download_button(
                            label="Download Complete Report (Excel)",
                            data=excel_data,
                            file_name=f"cbtad_results_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                        # 清理临时文件
                        os.remove(excel_file)

                    else:  # JSON
                        # 准备JSON导出
                        export_data = {
                            "metadata": {
                                "timestamp": timestamp,
                                "task_type": task_type,
                                "error_rate": st.session_state['error_rate'],
                                "error_composition": {
                                    "missing": st.session_state['missing_ratio'],
                                    "outliers": st.session_state['outlier_ratio'],
                                    "noise": st.session_state['noise_ratio']
                                }
                            },
                            "results": {
                                "performance_summary": summary_data,
                                "action_summary": action_summary,
                                "model_scores": {model: scores for model, scores in model_scores.items()},
                                "tolerance_thresholds": tolerance_data,
                                "best_model": best_model
                            }
                        }

                        # 转换为JSON
                        json_str = json.dumps(export_data, indent=2)

                        st.download_button(
                            label="Download Results (JSON)",
                            data=json_str,
                            file_name=f"cbtad_results_{timestamp}.json",
                            mime="application/json",
                        )

                    st.success("Export prepared successfully!")

        with col2:
            st.markdown("""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;text-align:center;">
                <h4 style="margin:0;color:#666;">Visualizations Export</h4>
                <p style="margin:10px 0;">Export generated charts and visualizations</p>
            </div>
            """, unsafe_allow_html=True)

            # 查找当前所有可用的图像
            all_images = find_images(task_type, error_rate)

            if all_images:
                st.write(f"Found {len(all_images)} visualization images.")

                if st.checkbox("Preview visualizations", value=False):
                    # 创建一个图片预览网格
                    preview_cols = st.columns(1)
                    for i, img_path in enumerate(all_images[:4]):  # 限制预览数量
                        with preview_cols[0]:
                            img = Image.open(img_path)
                            caption = os.path.basename(img_path)
                            st.image(img, caption=caption, use_container_width=True)

                    if len(all_images) > 4:
                        st.info(f"+ {len(all_images) - 4} more visualizations available for download")
                download_cols = st.columns(2)
                with download_cols[0]:
                    if st.button("Export All Visualizations"):
                        # 创建一个临时ZIP文件
                        import zipfile

                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        zip_file = f"cbtad_visualizations_{timestamp}.zip"
                        with zipfile.ZipFile(zip_file, 'w') as zipf:
                            for img in all_images:
                                zipf.write(img, arcname=os.path.basename(img))

                        # 读取ZIP文件并提供下载
                        with open(zip_file, "rb") as f:
                            zip_data = f.read()
                        st.download_button(
                            label="Download All Visualizations (ZIP)",
                            data=zip_data,
                            file_name=f"cbtad_visualizations_{timestamp}.zip",
                            mime="application/zip",
                        )

                        # 清理临时文件
                        os.remove(zip_file)
                        st.success("Visualizations packaged successfully!")
                with download_cols[1]:
                    # 添加报告生成选项
                    if st.button("Generate PDF Report"):
                        with st.spinner("Generating visualization report..."):
                            # 使用matplotlib的PDF后端创建报告
                            try:
                                from matplotlib.backends.backend_pdf import PdfPages

                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                pdf_file = f"cbtad_report_{timestamp}.pdf"

                                with PdfPages(pdf_file) as pdf:
                                    # 创建标题页
                                    plt.figure(figsize=(8.5, 11))
                                    plt.text(0.5, 0.6, "CBTAD Visualization Report",
                                             ha='center', fontsize=24, weight='bold')
                                    plt.text(0.5, 0.5, f"Task Type: {task_type.upper()}",
                                             ha='center', fontsize=16)
                                    plt.text(0.5, 0.45, f"Error Rate: {error_rate:.2f}",
                                             ha='center', fontsize=16)
                                    plt.text(0.5, 0.4, f"Generated: {timestamp}",
                                             ha='center', fontsize=16)
                                    plt.text(0.5, 0.3, f"Best Model: {best_model.upper()}",
                                             ha='center', fontsize=20, color='#d62728')
                                    plt.axis('off')
                                    pdf.savefig()
                                    plt.close()

                                    # 添加所有找到的图像
                                    for img_path in all_images:
                                        img = Image.open(img_path)

                                        # 将PIL图像转换为matplotlib图像
                                        plt.figure(figsize=(8.5, 11))
                                        plt.imshow(img)
                                        plt.title(os.path.basename(img_path), fontsize=14)
                                        plt.axis('off')
                                        pdf.savefig()
                                        plt.close()

                                # 提供PDF下载
                                with open(pdf_file, "rb") as f:
                                    pdf_data = f.read()

                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_data,
                                    file_name=pdf_file,
                                    mime="application/pdf",
                                )

                                # 清理临时文件
                                os.remove(pdf_file)
                                st.success("Report generated successfully!")

                            except ImportError:
                                st.warning("PDF generation requires additional libraries.")
                                st.info("Please install matplotlib with PDF backend support.")
            else:
                st.info("No visualization images found. Generate visualizations in previous steps first.")

        with col3:
            st.markdown("""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;text-align:center;">
                <h4 style="margin:0;color:#666;">Model Deployment</h4>
                <p style="margin:10px 0;">Download trained models for deployment</p>
            </div>
            """, unsafe_allow_html=True)

            # 获取当前模型文件的列表
            model_files = glob.glob(f"dqn_agent_{task_type}_*.h5")

            if model_files:
                st.write(f"Found {len(model_files)} trained models:")

                # 创建模型列表和选择框
                model_options = [os.path.basename(file) for file in model_files]
                selected_model = st.selectbox("Select model to download:", model_options)

                # 显示所选模型的信息
                # 修复模型信息解析代码
                if selected_model:
                    model_path = os.path.join("", selected_model)

                    # 使用更健壮的方式解析模型文件名
                    import re

                    # 从文件名中提取任务类型和模型类型
                    model_name = selected_model.replace("dqn_agent_", "").replace(".h5", "")

                    # 尝试从文件名中提取错误率 (查找类似 0.30 的浮点数)
                    error_rate_match = re.search(r'(\d+\.\d+)', model_name)
                    model_error = float(error_rate_match.group(1)) if error_rate_match else "Multi-Rate"

                    # 提取任务类型 (classification 或 regression)
                    task_type = "classification" if "classification" in model_name else "regression"

                    # 提取模型类型 (移除任务类型和错误率后剩余的部分)
                    model_type = model_name.replace(f"{task_type}_", "")
                    if error_rate_match:
                        model_type = model_type.replace(error_rate_match.group(1), "").strip("_")

                    # 显示模型卡片
                    st.markdown(f"""
                    <div style="background-color:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px;">
                        <h5 style="margin:0;color:#333;">{selected_model}</h5>
                        <p style="margin:5px 0;font-size:0.9em;">
                            <span style="color:#666;">Task:</span> <b>{task_type}</b><br>
                            <span style="color:#666;">Model:</span> <b>{model_type}</b><br>
                            <span style="color:#666;">Error Rate:</span> <b>{model_error if isinstance(model_error, str) else f"{model_error:.2f}"}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 你的本地文件路径
                    file_path = "./" + selected_model

                    # 打开文件并创建下载按钮
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                        st.download_button(
                            label="📥 下载模型文件",
                            data=file_bytes,
                            file_name=model_name,  # 下载时保存的文件名
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            else:
                st.info("No trained models found. Complete the training step to generate models.")
