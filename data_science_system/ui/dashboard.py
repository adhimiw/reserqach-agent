"""
Logging Dashboard - Real-time monitoring of analysis execution
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List


def render_dashboard(analysis_log_path: str = None):
    """
    Render the logging dashboard
    
    Args:
        analysis_log_path: Path to execution log JSON file
    """
    st.set_page_config(
        page_title="Data Science Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Autonomous Data Science System Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    
    # Log file selector
    if analysis_log_path is None:
        analyses_dir = "output/analyses"
        if os.path.exists(analyses_dir):
            analyses = [d for d in os.listdir(analyses_dir) 
                       if os.path.isdir(os.path.join(analyses_dir, d))]
            
            if analyses:
                selected_analysis = st.sidebar.selectbox(
                    "Select Analysis",
                    analyses,
                    index=0
                )
                log_path = os.path.join(analyses_dir, selected_analysis, 
                                      "logs", "execution_log.json")
            else:
                st.sidebar.warning("No analyses found in output/analyses/")
                log_path = None
        else:
            st.sidebar.error("output/analyses/ directory not found")
            log_path = None
    else:
        log_path = analysis_log_path
    
    # Load log data
    if log_path and os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        
        execution_log = log_data.get('execution_log', [])
        results_summary = log_data.get('results_summary', {})
    else:
        st.error("No execution log found. Run an analysis first.")
        execution_log = []
        results_summary = {}
    
    # Main content
    if execution_log:
        # Summary metrics
        st.subheader("📈 Analysis Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Hypotheses", results_summary.get('hypotheses', 0))
        
        with col2:
            st.metric("Statistical Tests", results_summary.get('statistical_tests', 0))
        
        with col3:
            st.metric("Models Built", results_summary.get('models', 0))
        
        with col4:
            st.metric("Insights Generated", results_summary.get('insights', 0))
        
        with col5:
            st.metric("Visualizations", results_summary.get('visualizations', 0))
        
        st.markdown("---")
        
        # Execution timeline
        st.subheader("⏱️ Execution Timeline")
        
        # Create timeline dataframe
        timeline_data = []
        for entry in execution_log:
            timeline_data.append({
                "Step": entry['step'],
                "Status": entry['status'].upper(),
                "Message": entry.get('message', ''),
                "Time": datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Status coloring
        def color_status(status):
            if status == "START":
                return "🟢"
            elif status == "COMPLETE":
                return "✅"
            elif status == "ERROR":
                return "❌"
            return "⚪"
        
        timeline_df["Status"] = timeline_df["Status"].apply(color_status)
        
        st.dataframe(timeline_df, use_container_width=True)
        
        # Detailed log view
        st.subheader("📋 Detailed Execution Log")
        
        with st.expander("View Full Log", expanded=True):
            for i, entry in enumerate(execution_log, 1):
                status_emoji = color_status(entry['status'])
                
                st.markdown(f"""
                **Step {i}: {entry['step']}** {status_emoji}
                - **Status:** {entry['status']}
                - **Time:** {entry['timestamp']}
                - **Message:** {entry.get('message', 'N/A')}
                """)
                st.markdown("---")
        
        # Insights viewer
        st.subheader("💡 Key Insights")
        
        insights_path = os.path.dirname(log_path).replace('logs', 'insights')
        insights_file = os.path.join(insights_path, 'insights.json')
        
        if os.path.exists(insights_file):
            with open(insights_file, 'r') as f:
                insights = json.load(f)
            
            if insights:
                # Insight type filter
                insight_types = list(set(insight.get('type', 'unknown') for insight in insights))
                selected_types = st.multiselect(
                    "Filter by Insight Type",
                    insight_types,
                    default=insight_types
                )
                
                # Filter insights
                filtered_insights = [
                    insight for insight in insights 
                    if insight.get('type') in selected_types
                ]
                
                # Display insights
                for insight in filtered_insights[:10]:  # Show top 10
                    with st.expander(f"{insight.get('title', 'Untitled')}", expanded=False):
                        st.markdown(f"**What:** {insight.get('what', 'N/A')}")
                        st.markdown(f"**Why:** {insight.get('why', 'N/A')}")
                        st.markdown(f"**How:** {insight.get('how', 'N/A')}")
                        st.markdown(f"**Recommendation:** {insight.get('recommendation', 'N/A')}")
            else:
                st.info("No insights available yet.")
        else:
            st.info("Insights file not found.")
        
        # Visualizations viewer
        st.subheader("📊 Visualizations")
        
        viz_path = os.path.dirname(log_path).replace('logs', 'visualizations')
        
        if os.path.exists(viz_path):
            viz_files = [f for f in os.listdir(viz_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if viz_files:
                # Layout options
                cols = st.selectbox("Columns", [2, 3, 4], index=2)
                
                for i in range(0, len(viz_files), cols):
                    row_cols = st.columns(cols)
                    for j in range(cols):
                        if i + j < len(viz_files):
                            viz_file = viz_files[i + j]
                            with row_cols[j]:
                                st.image(os.path.join(viz_path, viz_file), 
                                          caption=viz_file, use_column_width=True)
            else:
                st.info("No visualizations available yet.")
        else:
            st.info("Visualizations directory not found.")
        
        # Model performance
        st.subheader("🎯 Model Performance")
        
        models_path = os.path.join(insights_path, 'models.json')
        
        if os.path.exists(models_path):
            with open(models_path, 'r') as f:
                models_data = json.load(f)
            
            if models_data.get('models'):
                # Create comparison table
                model_comparison = []
                
                for model_name, model_info in models_data['models'].items():
                    if isinstance(model_info, dict) and 'metrics' in model_info:
                        metrics = model_info['metrics']
                        
                        row = {
                            "Model": metrics.get('model_type', model_name),
                            "Type": models_data.get('task_type', 'unknown')
                        }
                        
                        # Add metrics based on task type
                        if 'test_accuracy' in metrics:
                            row['Accuracy'] = f"{metrics['test_accuracy']:.2%}"
                            row['F1-Score'] = f"{metrics['classification_report'].get('macro avg', {}).get('f1-score', 0):.3f}"
                        else:
                            row['R²'] = f"{metrics['test_r2']:.4f}"
                            row['RMSE'] = f"{metrics['test_mse'] ** 0.5:.4f}"
                        
                        model_comparison.append(row)
                
                if model_comparison:
                    st.dataframe(pd.DataFrame(model_comparison), use_container_width=True)
                    
                    # Feature importance
                    for model_name, model_info in models_data['models'].items():
                        if isinstance(model_info, dict) and 'metrics' in model_info:
                            metrics = model_info['metrics']
                            if 'feature_importance' in metrics:
                                st.markdown(f"**Feature Importance - {metrics.get('model_type', model_name)}**")
                                
                                feat_imp = pd.DataFrame(metrics['feature_importance'])
                                
                                if not feat_imp.empty:
                                    st.bar_chart(feat_imp.head(10).set_index('feature')['importance'])
                                else:
                                    st.info("No feature importance data available.")
                                
                                st.markdown("---")
            else:
                st.info("No model data available.")
        else:
            st.info("Model results not found.")
    
    # Auto-refresh option
    if st.sidebar.checkbox("Auto-refresh (30s)"):
        time.sleep(30)
        st.rerun()


def render_error_dashboard(error_log_path: str):
    """
    Render error log dashboard
    
    Args:
        error_log_path: Path to error log JSON file
    """
    st.set_page_config(
        page_title="Error Dashboard",
        page_icon="⚠️",
        layout="wide"
    )
    
    st.title("⚠️ Error & Recovery Dashboard")
    st.markdown("---")
    
    if error_log_path and os.path.exists(error_log_path):
        with open(error_log_path, 'r') as f:
            error_data = json.load(f)
        
        error_log = error_data.get('error_log', [])
        recovery_log = error_data.get('recovery_log', [])
        
        # Error statistics
        st.subheader("📊 Error Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Errors", len(error_log))
        
        with col2:
            recovery_rate = sum(1 for r in recovery_log if r.get('success', False)) / len(error_log) * 100 if error_log else 100
            st.metric("Recovery Rate", f"{recovery_rate:.1f}%")
        
        # Error types
        st.subheader("🔍 Error Types")
        
        error_types = {}
        for error in error_log:
            etype = error.get('error_type', 'Unknown')
            error_types[etype] = error_types.get(etype, 0) + 1
        
        if error_types:
            error_types_df = pd.DataFrame(list(error_types.items()), 
                                        columns=['Error Type', 'Count'])
            st.bar_chart(error_types_df.set_index('Error Type')['Count'])
        else:
            st.info("No errors recorded.")
        
        # Recovery strategies
        if recovery_log:
            st.subheader("🔧 Recovery Strategies")
            
            strategies = {}
            for recovery in recovery_log:
                strategy = recovery.get('recovery_strategy', 'unknown')
                strategies[strategy] = strategies.get(strategy, 0) + 1
            
            if strategies:
                strategies_df = pd.DataFrame(list(strategies.items()),
                                         columns=['Strategy', 'Count'])
                st.dataframe(strategies_df, use_container_width=True)
        
        # Detailed error log
        st.subheader("📋 Detailed Error Log")
        
        for i, error in enumerate(error_log, 1):
            with st.expander(f"Error {i}: {error.get('error_type', 'Unknown')}", expanded=False):
                st.code(f"""Function: {error.get('function', 'N/A')}
Attempt: {error.get('attempt', 1)}
Error Type: {error.get('error_type', 'Unknown')}
Message: {error.get('error_message', 'N/A')}

Traceback:
{error.get('traceback', 'N/A')}""", language="python")
        
        # Recovery log
        if recovery_log:
            st.subheader("✅ Recovery Log")
            
            for i, recovery in enumerate(recovery_log, 1):
                status_emoji = "✅" if recovery.get('success', False) else "❌"
                st.markdown(f"""
                **Recovery {i}** {status_emoji}
                - **Function:** {recovery.get('function', 'N/A')}
                - **Attempt:** {recovery.get('attempt', 1)}
                - **Strategy:** {recovery.get('recovery_strategy', 'N/A')}
                - **Success:** {recovery.get('success', False)}
                """)
    else:
        st.error("Error log file not found.")


def main():
    """Main entry point for dashboard"""
    st.sidebar.title("Dashboard Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Analysis Dashboard", "Error Dashboard"]
    )
    
    if page == "Analysis Dashboard":
        render_dashboard()
    elif page == "Error Dashboard":
        # Find latest error log
        analyses_dir = "output/analyses"
        if os.path.exists(analyses_dir):
            analyses = [d for d in os.listdir(analyses_dir) 
                       if os.path.isdir(os.path.join(analyses_dir, d))]
            
            if analyses:
                # Find most recent
                latest_analysis = max(analyses, 
                                   key=lambda a: os.path.getmtime(os.path.join(analyses_dir, a)))
                error_log_path = os.path.join(analyses_dir, latest_analysis,
                                           "logs", "error_log.json")
                
                if not os.path.exists(error_log_path):
                    error_log_path = None
            else:
                error_log_path = None
        else:
            error_log_path = None
        
        render_error_dashboard(error_log_path)


if __name__ == "__main__":
    main()
