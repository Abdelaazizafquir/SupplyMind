"""
Intelligent Supply Chain PFE Dashboard
========================================
A Streamlit interactive dashboard to visualize MAPPO+GNN training,
baseline comparisons, disruption resilience, and LLM Orchestrator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

st.set_page_config(
    page_title="Intelligent Supply Chain Dashboard",
    page_icon="🌍",
    layout="wide",
)

# Project root
dashboard_dir = Path(__file__).parent.resolve()
project_root = dashboard_dir.parent

st.sidebar.title("Intelligent SC PFE")
st.sidebar.markdown("""
**GNN-Enhanced MAPPO + LLM Coordinator**
for Multi-Echelon Supply Chain Optimization
""")

page = st.sidebar.radio("Navigation", [
    "1. Agent Training",
    "2. Baselines Comparison",
    "3. Disruption Resilience",
    "4. LLM Orchestrator",
])


def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# ==========================================
# PAGE 1: MAPPO TRAINING
# ==========================================
if page == "1. Agent Training":
    st.title("🧠 MAPPO+GNN Training Progress")
    st.markdown("Training of the GNN-enhanced Multi-Agent PPO on Kaggle (1500 episodes).")

    training_results_path = project_root / "experiments" / "mappo_training" / "training_results.json"
    training_curve_path = project_root / "experiments" / "mappo_training" / "training_curve.png"
    model_path = project_root / "models" / "agents" / "mappo_gnn_final.pt"

    training_data = load_json(training_results_path)

    if training_data:
        # Parse eval results (stored as string repr of list of dicts)
        eval_results = training_data.get("eval_results", [])
        if isinstance(eval_results, str):
            try:
                eval_results = json.loads(eval_results.replace("'", '"'))
            except Exception:
                eval_results = []

        config = training_data.get("config", {})
        if isinstance(config, str):
            try:
                config = json.loads(config.replace("'", '"'))
            except Exception:
                config = {}

        # ── Key metrics ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Episodes", training_data.get("total_episodes", "N/A"))
        col2.metric("Training Time", f"{training_data.get('training_time_seconds', 0)/60:.0f} min")

        if eval_results:
            best = max(eval_results, key=lambda x: x.get("avg_fill_rate", 0))
            final = eval_results[-1]
            col3.metric("Best Fill Rate", f"{best['avg_fill_rate']:.1%}",
                        delta=f"Ep {best.get('episode', '?')}")
            col4.metric("Final Fill Rate", f"{final['avg_fill_rate']:.1%}")

        # ── Training curve image ──
        if training_curve_path.exists():
            st.subheader("Training Curves")
            st.image(str(training_curve_path), use_container_width=True)

        # ── Eval results as interactive charts ──
        if eval_results:
            st.subheader("Evaluation Every 100 Episodes")
            eval_df = pd.DataFrame(eval_results)

            tab_fill, tab_cost, tab_reward = st.tabs(["Fill Rate", "Total Cost", "Reward"])
            with tab_fill:
                st.line_chart(eval_df.set_index("episode")["avg_fill_rate"],
                              y_label="Fill Rate", x_label="Episode")
                st.caption("Fill rate convergence: exploration valley at ep 200-400, then recovery to >97%.")
            with tab_cost:
                st.line_chart(eval_df.set_index("episode")["avg_cost"],
                              y_label="Avg Cost ($)", x_label="Episode")
            with tab_reward:
                st.line_chart(eval_df.set_index("episode")["avg_reward"],
                              y_label="Avg Reward", x_label="Episode")

        # ── Config ──
        with st.expander("Model Configuration"):
            st.json(config)

        # ── Model checkpoint info ──
        if model_path.exists():
            st.success(f"Trained model available: `models/agents/{model_path.name}`")
        else:
            st.warning("Model file not found in `models/agents/`. Download from Kaggle.")
    else:
        st.warning("No training results found. Run training on Kaggle and place `training_results.json` in `experiments/mappo_training/`.")


# ==========================================
# PAGE 2: BASELINES COMPARISON
# ==========================================
elif page == "2. Baselines Comparison":
    st.title("📊 MAPPO+GNN vs Classical Baselines")
    st.markdown("Head-to-head comparison on the same normalized demand environment (10 episodes each).")

    comparison_path = project_root / "experiments" / "final_comparison" / "comparison_results.json"
    comparison_img = project_root / "experiments" / "final_comparison" / "full_comparison.png"

    data = load_json(comparison_path)

    if data:
        # ── Summary metrics ──
        policies = list(data.keys())
        rows = []
        for pname in policies:
            m = data[pname]["metrics"]
            rows.append({
                "Policy": pname,
                "Total Cost ($)": f"${m['total_cost']['mean']:,.0f}",
                "Fill Rate": f"{m['fill_rate']['mean']:.1%}",
                "Stockout Rate": f"{m['stockout_rate']['mean']:.2%}",
                "Bullwhip Ratio": f"{m['bullwhip_ratio']['mean']:.1f}",
                "Cost (raw)": m['total_cost']['mean'],
                "Fill (raw)": m['fill_rate']['mean'],
            })
        df = pd.DataFrame(rows)

        # Highlight MAPPO row
        st.subheader("Summary Table")
        display_df = df[["Policy", "Total Cost ($)", "Fill Rate", "Stockout Rate", "Bullwhip Ratio"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Key comparison
        mappo = data.get("MAPPO+GNN", {}).get("metrics", {})
        best_baseline_name = min(
            [p for p in policies if p != "MAPPO+GNN"],
            key=lambda p: data[p]["metrics"]["total_cost"]["mean"]
        )
        best_bl = data[best_baseline_name]["metrics"]

        col1, col2, col3 = st.columns(3)
        col1.metric("MAPPO Fill Rate", f"{mappo['fill_rate']['mean']:.1%}",
                     delta=f"+{(mappo['fill_rate']['mean'] - best_bl['fill_rate']['mean']):.1%} vs {best_baseline_name}")
        col2.metric("MAPPO Cost", f"${mappo['total_cost']['mean']:,.0f}")
        col3.metric(f"{best_baseline_name} Cost", f"${best_bl['total_cost']['mean']:,.0f}",
                     delta="Lowest baseline")

        # ── Full comparison chart ──
        if comparison_img.exists():
            st.subheader("4-Panel Comparison")
            st.image(str(comparison_img), use_container_width=True)

        # ── Interactive charts ──
        st.subheader("Interactive Comparison")
        tab_cost, tab_fill = st.tabs(["Cost Breakdown", "Fill Rate"])
        with tab_cost:
            cost_df = pd.DataFrame({
                "Policy": [r["Policy"] for r in rows],
                "Total Cost": [r["Cost (raw)"] for r in rows],
            }).set_index("Policy")
            st.bar_chart(cost_df)
        with tab_fill:
            fill_df = pd.DataFrame({
                "Policy": [r["Policy"] for r in rows],
                "Fill Rate": [r["Fill (raw)"] for r in rows],
            }).set_index("Policy")
            st.bar_chart(fill_df)

        st.info("""
        **Key insight:** MAPPO+GNN achieves the highest fill rate (99.1%) requiring zero manual parameter tuning.
        Classical baselines need careful parameter selection — wrong parameters lead to catastrophic performance (e.g. Base-Stock 200 at 78.8%).
        """)
    else:
        st.warning("No comparison results. Run `python notebooks/05_full_comparison.py`.")


# ==========================================
# PAGE 3: DISRUPTION RESILIENCE
# ==========================================
elif page == "3. Disruption Resilience":
    st.title("🛡️ Disruption Resilience Analysis")
    st.markdown("Testing how each policy handles demand spikes, supplier failures, and combined crises.")

    disruption_path = project_root / "experiments" / "disruption_comparison" / "disruption_results.json"
    disruption_img = project_root / "experiments" / "disruption_comparison" / "disruption_comparison.png"

    data = load_json(disruption_path)

    if data:
        scenarios = list(data.keys())
        policies = list(data[scenarios[0]].keys()) if scenarios else []

        # ── Degradation table ──
        st.subheader("Resilience Score: Fill Rate Degradation")
        if "Normal (Baseline)" in data:
            deg_rows = []
            for pname in policies:
                normal_fill = data["Normal (Baseline)"][pname]["metrics"]["fill_rate"]["mean"]
                row = {"Policy": pname}
                for scenario in scenarios:
                    if scenario == "Normal (Baseline)":
                        row[scenario] = f"{data[scenario][pname]['metrics']['fill_rate']['mean']:.1%}"
                    else:
                        disrupted_fill = data[scenario][pname]["metrics"]["fill_rate"]["mean"]
                        degradation = (normal_fill - disrupted_fill) / max(normal_fill, 1e-8) * 100
                        row[scenario] = f"{degradation:.1f}% drop"
                deg_rows.append(row)
            deg_df = pd.DataFrame(deg_rows)
            st.dataframe(deg_df, use_container_width=True, hide_index=True)

        # ── Full summary table ──
        st.subheader("Full Results by Scenario")
        for scenario in scenarios:
            with st.expander(f"📋 {scenario}"):
                rows = []
                for pname in policies:
                    m = data[scenario][pname]["metrics"]
                    rows.append({
                        "Policy": pname,
                        "Cost ($)": f"${m['total_cost']['mean']:,.0f}",
                        "Fill Rate": f"{m['fill_rate']['mean']:.1%}",
                        "Stockout": f"{m['stockout_rate']['mean']:.2%}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Key metrics ──
        if "Combined Crisis" in data and "MAPPO+GNN" in data["Combined Crisis"]:
            st.subheader("Combined Crisis Performance")
            col1, col2, col3 = st.columns(3)
            for i, pname in enumerate(policies[:3]):
                m = data["Combined Crisis"][pname]["metrics"]
                [col1, col2, col3][i].metric(
                    pname,
                    f"{m['fill_rate']['mean']:.1%}",
                    delta=f"${m['total_cost']['mean']:,.0f}",
                    delta_color="off")

        # ── Disruption chart ──
        if disruption_img.exists():
            st.subheader("4-Panel Disruption Analysis")
            st.image(str(disruption_img), use_container_width=True)

        st.success("""
        **Key finding:** MAPPO+GNN degrades only 3.4% under combined crisis while classical baselines
        lose 17-20% fill rate. The agent adapts its ordering dynamically instead of following fixed rules.
        """)
    else:
        st.warning("No disruption results. Run `python notebooks/06_disruption_comparison.py`.")


# ==========================================
# PAGE 4: LLM ORCHESTRATOR
# ==========================================
elif page == "4. LLM Orchestrator":
    st.title("🤖 LLM Coordinator Simulator")
    st.markdown("Step through a supply chain simulation and watch the LLM Coordinator interpret MAPPO decisions in real-time.")

    import sys
    sys.path.insert(0, str(project_root))

    # ── Sidebar Controls ──
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Simulator Config")
    llm_provider = st.sidebar.selectbox("LLM Provider", ["mock", "ollama", "openai"], index=0,
        help="'mock' uses pre-written responses. 'ollama' uses a local model. 'openai' requires an API key.")

    if llm_provider == "openai":
        openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
        os.environ["OPENAI_API_KEY"] = openai_key or ""

    if llm_provider == "ollama":
        ollama_model = st.sidebar.text_input("Ollama Model Name", value="llama3:8b")
    else:
        ollama_model = "llama3:8b"

    trigger_disruption = st.sidebar.toggle("💥 Inject Factory Fire (Day 20+)", value=False)

    # ── Session State ──
    for key in ["sim_env", "sim_wrapper", "sim_gnn", "sim_actor", "sim_roles",
                "sim_static", "sim_adj", "sim_obs", "sim_step", "sim_log", "sim_done"]:
        if key not in st.session_state:
            st.session_state[key] = None if key not in ["sim_step", "sim_log", "sim_done"] else (0 if key == "sim_step" else ([] if key == "sim_log" else False))

    col_init, col_step, col_reset = st.columns([1, 1, 1])

    # ── INIT ──
    with col_init:
        if st.button("🔌 Initialize Simulator", use_container_width=True):
            try:
                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                from torch.distributions import Normal
                import warnings
                from src.environment.supply_chain_env import MultiEchelonInventoryEnv
                from src.environment.disruption_wrapper import DisruptionWrapper

                model_path = project_root / "models" / "agents" / "mappo_gnn_final.pt"
                if not model_path.exists():
                    st.error("No model found at `models/agents/mappo_gnn_final.pt`!")
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        chkpt = torch.load(model_path, map_location="cpu", weights_only=False)

                    config = chkpt.get("config", {})
                    HIDDEN_DIM = config.get("hidden_dim", 512)
                    OBS_DIM = 12
                    GNN_OUT = 64
                    NUM_AGENTS = 6

                    # Build inline models matching Kaggle architecture
                    class GCNLayer(nn.Module):
                        def __init__(self, in_d, out_d):
                            super().__init__()
                            self.linear = nn.Linear(in_d, out_d)
                        def forward(self, x, adj):
                            adj_hat = adj + torch.eye(adj.size(0))
                            deg = adj_hat.sum(1).clamp(min=1)
                            norm = deg.pow(-0.5).unsqueeze(1) * adj_hat * deg.pow(-0.5).unsqueeze(0)
                            return torch.mm(norm, self.linear(x))

                    class GNNComm(nn.Module):
                        def __init__(self, inp, hid=64, out=32, nl=2):
                            super().__init__()
                            self.input_proj = nn.Linear(inp, hid)
                            self.layers = nn.ModuleList()
                            self.norms = nn.ModuleList()
                            for i in range(nl):
                                self.layers.append(GCNLayer(hid, hid if i < nl-1 else out))
                                self.norms.append(nn.LayerNorm(hid if i < nl-1 else out))
                        def forward(self, x, adj):
                            h = F.relu(self.input_proj(x))
                            for i, (l, n) in enumerate(zip(self.layers, self.norms)):
                                hp = h; h = n(l(h, adj))
                                if i < len(self.layers)-1:
                                    h = F.relu(h)
                                    if h.shape == hp.shape: h = h + hp
                            return h

                    class Actor(nn.Module):
                        def __init__(self, obs_d, hid):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(obs_d+3, hid), nn.LayerNorm(hid), nn.ReLU(),
                                nn.Linear(hid, hid), nn.LayerNorm(hid), nn.ReLU())
                            self.mean_head = nn.Linear(hid, 1)
                            self.log_std = nn.Parameter(torch.zeros(1))
                        def forward(self, obs, role):
                            x = torch.cat([obs, role], -1)
                            return torch.sigmoid(self.mean_head(self.net(x)))

                    # Graph
                    adj = np.zeros((6, 6), dtype=np.float32)
                    for r in range(3):
                        d = 3 + (r % 2); adj[r,d]=adj[d,r]=1.0
                    for d in range(2):
                        s = 5; adj[3+d,s]=adj[s,3+d]=1.0

                    static = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],
                                       [0,1,0,.5],[0,1,0,.5],[0,0,1,1]], dtype=np.float32)
                    roles = np.array([[1,0,0],[1,0,0],[1,0,0],
                                     [0,1,0],[0,1,0],[0,0,1]], dtype=np.float32)

                    gnn = GNNComm(OBS_DIM + 4, 64, GNN_OUT, 2)
                    actor = Actor(OBS_DIM + GNN_OUT, HIDDEN_DIM)
                    gnn.load_state_dict(chkpt["gnn"])
                    actor.load_state_dict(chkpt["actor"])
                    gnn.eval(); actor.eval()

                    # Environment
                    env_config = {"num_retailers": 3, "num_distributors": 2, "num_suppliers": 1,
                                  "num_periods": 100, "holding_cost": 1.0, "ordering_cost": 50.0,
                                  "shortage_cost": 500.0, "initial_inventory": 100, "max_order_quantity": 500}
                    env = MultiEchelonInventoryEnv(env_config)

                    # Generate demand
                    np.random.seed(42)
                    t_arr = np.arange(100)
                    dem = np.zeros((100, 3))
                    for r in range(3):
                        b = 25.0 * (1 + 0.2*np.random.random())
                        dem[:,r] = np.maximum(0, b + 3*np.sin(2*np.pi*t_arr/7) + np.random.normal(0,5,100))
                    env.set_demand_data(dem)

                    obs, _ = env.reset(seed=42)

                    st.session_state.sim_env = env
                    st.session_state.sim_gnn = gnn
                    st.session_state.sim_actor = actor
                    st.session_state.sim_roles = torch.tensor(roles, dtype=torch.float32)
                    st.session_state.sim_static = torch.tensor(static, dtype=torch.float32)
                    st.session_state.sim_adj = torch.tensor(adj)
                    st.session_state.sim_obs = obs
                    st.session_state.sim_step = 0
                    st.session_state.sim_log = []
                    st.session_state.sim_done = False
                    st.success(f"Simulator initialized with Kaggle-trained model (hidden_dim={HIDDEN_DIM})")
            except Exception as e:
                st.error(f"Initialization failed: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── STEP ──
    with col_step:
        if st.button("▶️ Step 1 Day", use_container_width=True,
                      disabled=st.session_state.sim_env is None or st.session_state.sim_done):
            try:
                import torch
                import warnings

                env = st.session_state.sim_env
                gnn = st.session_state.sim_gnn
                actor = st.session_state.sim_actor
                roles_t = st.session_state.sim_roles
                static_t = st.session_state.sim_static
                adj_t = st.session_state.sim_adj
                obs = st.session_state.sim_obs
                step = st.session_state.sim_step

                # Build 12-dim Kaggle-format observations
                nr = env.num_retailers
                last_dem = np.zeros(nr)
                if env.episode_data["demand_history"]:
                    last_d = env.episode_data["demand_history"][-1]
                    if isinstance(last_d, (list, np.ndarray)):
                        last_dem = np.array(last_d[:nr])

                total_rd = float(np.sum(last_dem))
                obs_list = []
                for i, aid in enumerate(env.agent_ids):
                    node = env.nodes[i]
                    pipe = sum(q for _, q in node.pipeline)
                    ld = last_dem[i]/100 if i < nr else total_rd/(nr*100)
                    cap = 1000 if i < nr else (2500 if i < nr+2 else 5000)
                    obs_list.append(torch.tensor([
                        node.inventory/1000, pipe/1000, ld,
                        (node.inventory+pipe)/cap, 0.0,
                        step/100, 1 if i<nr else 0, 1 if nr<=i<nr+2 else 0,
                        1 if i>=nr+2 else 0, 0.0,
                        total_rd/(nr*100), len([p for p in node.pipeline if p[1]>0])/5
                    ], dtype=torch.float32))

                obs_t = torch.stack(obs_list)
                gnn_in = torch.cat([obs_t, static_t], dim=-1)

                with torch.no_grad():
                    gnn_out = gnn(gnn_in, adj_t)
                    aug = torch.cat([obs_t, gnn_out], dim=-1)

                agent_actions = {}
                for i, aid in enumerate(env.agent_ids):
                    mean = actor(aug[i:i+1], roles_t[i:i+1])
                    a_kaggle = float(mean[0, 0])
                    # Convert: Kaggle linear → local squared
                    a_local = np.sqrt(max(0, a_kaggle))
                    agent_actions[aid] = np.array([a_local], dtype=np.float32)

                # Disruption info for LLM
                disruption_info = {"disruption_active": False}
                if trigger_disruption and step >= 20:
                    disruption_info = {
                        "disruption_active": True,
                        "disruption_severity": 2.5,
                        "failed_nodes": ["distributor_0"],
                        "disruption_types": ["factory_fire"],
                    }

                # Try LLM coordinator
                coord_explanation = ""
                coord_mode = "direct"
                try:
                    from src.coordinator.llm_coordinator import LLMCoordinatorGraph
                    env_state = {aid: {"inventory": float(env.nodes[i].inventory),
                                       "capacity": float(env.nodes[i].capacity),
                                       "node_type": env.nodes[i].node_type}
                                 for i, aid in enumerate(env.agent_ids)}
                    coordinator = LLMCoordinatorGraph(config={
                        "llm_provider": llm_provider,
                        "llm_model": ollama_model,
                        "advisory_enabled": True,
                        "override_enabled": disruption_info.get("disruption_active", False),
                        "explanation_enabled": True,
                    })
                    agent_actions, coord_info = coordinator.coordinate(
                        step=step, observations=obs, proposed_actions=agent_actions,
                        env_state=env_state, disruption_info=disruption_info,
                        agent_ids=env.agent_ids,
                        num_agents_config={"num_retailers": env.num_retailers,
                                           "num_distributors": env.num_distributors,
                                           "num_suppliers": env.num_suppliers},
                    )
                    coord_explanation = coord_info.get("explanation", "")
                    coord_mode = str(coord_info.get("mode", "advisory"))
                except Exception:
                    coord_explanation = "LLM coordinator not available. Using raw MAPPO actions."
                    coord_mode = "direct (no LLM)"

                # Step env
                next_obs, rewards, terminated, truncated, infos = env.step(agent_actions)
                total_cost = sum(-float(r) for r in rewards.values())

                log_entry = {
                    "step": step + 1,
                    "cost": round(total_cost, 2),
                    "mode": coord_mode,
                    "disruption": disruption_info.get("disruption_active", False),
                    "explanation": coord_explanation,
                    "inventories": {aid: round(float(env.nodes[i].inventory), 1)
                                    for i, aid in enumerate(env.agent_ids)},
                    "orders": {aid: round(float(agent_actions[aid][0]**2 * env.nodes[i].capacity), 1)
                               for i, aid in enumerate(env.agent_ids)},
                }
                st.session_state.sim_log.append(log_entry)
                st.session_state.sim_obs = next_obs
                st.session_state.sim_step = step + 1

                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    st.session_state.sim_done = True

            except Exception as e:
                st.error(f"Simulation step failed: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── RESET ──
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            for key in ["sim_env", "sim_wrapper", "sim_gnn", "sim_actor", "sim_roles",
                        "sim_static", "sim_adj", "sim_obs", "sim_step", "sim_log", "sim_done"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── LOG DISPLAY ──
    st.divider()
    if st.session_state.sim_log:
        st.subheader(f"📋 Simulation Log — Day {st.session_state.sim_step}")

        latest = st.session_state.sim_log[-1]
        mode_color = "🔴" if latest["disruption"] else "🟢"
        st.markdown(f"**{mode_color} Day {latest['step']} | Mode: `{latest['mode']}` | Step Cost: `${latest['cost']:,.2f}`**")

        col_inv, col_ord = st.columns(2)
        with col_inv:
            st.markdown("**🏭 Inventory Levels**")
            inv_df = pd.DataFrame.from_dict(latest["inventories"], orient="index", columns=["Inventory"])
            st.dataframe(inv_df, use_container_width=True)
        with col_ord:
            st.markdown("**📦 Orders Placed**")
            ord_df = pd.DataFrame.from_dict(latest["orders"], orient="index", columns=["Order Qty"])
            st.dataframe(ord_df, use_container_width=True)

        st.markdown("### 🧠 LLM Coordinator Decision")
        st.info(latest["explanation"])

        # Cumulative cost chart
        st.divider()
        st.subheader("📈 Cumulative Cost Trajectory")
        cost_series = pd.DataFrame({
            "Day": [e["step"] for e in st.session_state.sim_log],
            "Step Cost ($)": [e["cost"] for e in st.session_state.sim_log]
        })
        st.line_chart(cost_series.set_index("Day"))

        with st.expander("📜 Full Explanation Log"):
            for entry in reversed(st.session_state.sim_log):
                disrupt_tag = " 💥 DISRUPTION" if entry["disruption"] else ""
                st.markdown(f"**Day {entry['step']}{disrupt_tag}** — *{entry['mode']}*")
                st.write(entry["explanation"])
                st.divider()
    else:
        st.info("Click **Initialize Simulator** then **Step 1 Day** to begin.")
        st.markdown("""
        ### How This Works
        1. **Initialize** loads the Kaggle-trained MAPPO+GNN model
        2. **Step 1 Day** runs the neural network to generate orders, then passes them to the **LLM Coordinator** which:
           - Checks if an **Emergency Override** is needed 🔴
           - Or confirms the MAPPO decisions are sound 🟢
        3. Enable **💥 Inject Factory Fire** in the sidebar to test crisis mode!
        """)

