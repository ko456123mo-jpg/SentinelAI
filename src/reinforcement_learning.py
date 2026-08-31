"""
SentinelAI - Stage 10: Reinforcement Learning (from-scratch Q-Learning)
=======================================================================
NO RL library is used - the Q-learning update is implemented by hand so
the learning process can be fully explained (project requirement).

THE PROBLEM
-----------
When SentinelAI classifies an event, someone must choose the RESPONSE:
    Actions : [Monitor, Alert_Analyst, Block_IP, Isolate_Host]
based on
    State   : threat_level (Benign/Suspicious/Malicious)
              confidence    (Low/Medium/High)
              asset         (Low/High criticality)
              -> 3 x 3 x 2 = 18 discrete states

THE ENVIRONMENT (SecurityResponseEnv)
-------------------------------------
step(action) returns a reward from an explicit SOC cost model:

  * Missing a real attack is the WORST outcome:
        reward -= 4 x threat_level        (Monitor on Malicious = -8)
  * Over-reacting on benign traffic annoys users / wastes analyst time:
        Block on benign  = -3, Isolate on benign = -5
  * Doing nothing sensible on benign traffic: +1 (Monitor)
  * Correct escalation earns points:
        Alert on threat = +2, Block on threat = +3,
        Isolate on a HIGH-criticality Malicious host = +5 (best move),
        Isolate on a low-criticality/suspicious host = +1 (overkill).

Q-LEARNING (tabular, epsilon-greedy)
------------------------------------
    Q[s,a] <- Q[s,a] + alpha * ( r + gamma * max_a' Q[s',a'] - Q[s,a] )
    epsilon decays 1.0 -> 0.05 so exploration gradually turns into
    exploitation of the learned policy.

The agent therefore LEARNS FROM REWARDS: behaviours that reduced the SOC
cost are reinforced, behaviours that were punished are avoided - no
labels, no supervised signal.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src import visualization as viz
from src.config import StageTimer, save_json

N_STATES = (len(config.THREAT_LEVELS) *
            len(config.CONF_LEVELS) *
            len(config.CRITICALITY_LEVELS))          # 18
N_ACTIONS = len(config.RL_ACTIONS)                   # 4

# ---- cost model (documented above) -------------------------------------
REWARD = {
    "miss_scale": 4,          # -4 x threat_level when action too soft
    "benign_monitor": +1,
    "benign_block": -3,
    "benign_isolate": -5,
    "benign_alert": -1,
    "threat_alert": +2,
    "threat_block": +3,
    "threat_isolate_high_crit_malicious": +5,
    "threat_isolate_other": +1,      # overkill
    "threat_monitor_suspicious": -2, # suspicious at least deserves alert
}


class SecurityResponseEnv:
    """Tiny custom environment (states, actions, rewards, transitions)."""

    def __init__(self, seed=config.SEED):
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self):
        self.state = int(self.rng.integers(N_STATES))
        return self.state

    @staticmethod
    def decode(s):
        """state id -> (threat, conf, crit) indices."""
        threat = s // 6
        conf = (s % 6) // 2
        crit = s % 2
        return threat, conf, crit

    @staticmethod
    def encode(threat, conf, crit):
        return threat * 6 + conf * 2 + crit

    def step(self, action):
        threat, conf, crit = self.decode(self.state)
        r = self._reward(threat, crit, action)
        nxt = self.reset()                 # episodic one-shot transitions
        return nxt, float(r)

    @staticmethod
    def _reward(threat, crit, action):
        if threat == 0:                                   # BENIGN traffic
            return {0: REWARD["benign_monitor"],          # Monitor
                    1: REWARD["benign_alert"],            # Alert
                    2: REWARD["benign_block"],            # Block
                    3: REWARD["benign_isolate"]}[action]  # Isolate
        # ------- real threat (suspicious or malicious) -------
        if action == 0:                                   # too soft
            return -REWARD["miss_scale"] * threat
        if action == 1:                                   # escalate
            return REWARD["threat_alert"]
        if action == 2:
            return REWARD["threat_block"]
        # isolate
        if threat == 2 and crit == 1:
            return REWARD["threat_isolate_high_crit_malicious"]
        return REWARD["threat_isolate_other"]


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration (from scratch)."""

    def __init__(self, alpha=config.RL_ALPHA, gamma=config.RL_GAMMA):
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha, self.gamma = alpha, gamma

    def act(self, s, epsilon=0.0):
        if np.random.random() < epsilon:                 # explore
            return int(np.random.randint(N_ACTIONS))
        return int(np.argmax(self.Q[s]))                 # exploit

    def learn(self, s, a, r, s2):
        """The Q-learning update rule - implemented manually."""
        td_target = r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])


def state_label(s):
    t, c, cr = SecurityResponseEnv.decode(s)
    return (f"{config.THREAT_LEVELS[t][:4]}/{config.CONF_LEVELS[c]}/"
            f"{'HC' if cr else 'LC'}")


def run() -> dict:
    with StageTimer(10, "Reinforcement Learning (Q-Learning response policy)"):
        config.ensure_dirs()
        env = SecurityResponseEnv()
        agent = QLearningAgent()

        eps = config.RL_EPS_START
        rewards_hist, eps_hist = [], []
        for ep in range(config.RL_EPISODES):
            s = env.reset()
            a = agent.act(s, epsilon=eps)
            s2, r = env.step(a)
            agent.learn(s, a, r, s2)      # <- learning FROM the reward
            rewards_hist.append(r)
            eps_hist.append(eps)
            eps = max(config.RL_EPS_END, eps * config.RL_EPS_DECAY)

        # moving-average learning curve
        window = 150
        ma = pd.Series(rewards_hist).rolling(window, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.plot(rewards_hist, alpha=.25, color="#94a3b8", label="episode reward")
        ax.plot(ma, color=viz.PALETTE[0], linewidth=2.2,
                label=f"{window}-episode moving average")
        ax.set_xlabel("episode"); ax.set_ylabel("reward")
        ax.set_title("Q-Learning agent - reward over training\n"
                     "(learns to avoid costly mistakes without labels)")
        ax.legend()
        viz.save_fig(fig, "rl_learning_curve.png")

        fig = viz.line_chart(range(config.RL_EPISODES), [eps_hist],
                             ["epsilon"], "Exploration decay (epsilon-greedy)",
                             "episode", "epsilon")
        viz.save_fig(fig, "rl_epsilon_decay.png")

        # learned Q-table heatmap
        fig, ax = plt.subplots(figsize=(7.6, 8.2))
        im = ax.imshow(agent.Q, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(N_ACTIONS))
        ax.set_xticklabels(config.RL_ACTIONS, rotation=20, ha="right")
        ax.set_yticks(range(N_STATES))
        ax.set_yticklabels([state_label(s) for s in range(N_STATES)],
                           fontsize=8)
        best = agent.Q.argmax(1)
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                txt = f"{agent.Q[s, a]:.1f}"
                ax.text(a, s, txt, ha="center", va="center", fontsize=7,
                        fontweight="bold" if a == best[s] else "normal",
                        color="black")
        ax.set_title("Learned Q-table (bold = chosen action)\n"
                     "state = threat/confidence/asset-criticality")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.03)
        viz.save_fig(fig, "rl_qtable_heatmap.png")

        # ---- evaluate policy vs baselines over the full state grid ------
        def policy_reward(agent_fn):
            total, n = 0.0, 0
            for s in range(N_STATES):
                for _ in range(300):
                    env.state = s
                    a = agent_fn(s)
                    _, r = env.step(a)
                    total += r; n += 1
            return total / n

        env2 = SecurityResponseEnv(seed=7)
        evals = {
            "AlwaysMonitor": policy_reward(lambda s: 0),
            "AlwaysBlock": policy_reward(lambda s: 2),
            "Random": policy_reward(lambda s:
                                    int(env2.rng.integers(N_ACTIONS))),
            "Q-Learning policy": policy_reward(lambda s: agent.act(s)),
        }
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.bar(evals.keys(), evals.values(),
               color=[viz.PALETTE[4], viz.PALETTE[1], viz.PALETTE[3],
                      viz.PALETTE[2]])
        ax.bar_label(ax.containers[0], fmt="%.2f")
        ax.set_ylabel("average reward per decision")
        ax.set_title("Learned RL policy vs baseline strategies")
        ax.tick_params(axis="x", rotation=12)
        viz.save_fig(fig, "rl_vs_baselines.png")

        # export the learned policy table
        pol = pd.DataFrame(agent.Q, columns=config.RL_ACTIONS)
        pol.index = [state_label(s) for s in range(N_STATES)]
        pol["POLICY"] = [config.RL_ACTIONS[a] for a in agent.Q.argmax(1)]
        pol.to_csv(os.path.join(config.MODEL_DIR, "rl_policy.csv"))

        joblib.dump({"Q": agent.Q,
                     "actions": config.RL_ACTIONS},
                    os.path.join(config.MODEL_DIR, "rl_qtable.joblib"))

        summary = {
            "algorithm": "tabular Q-Learning (hand-implemented update rule)",
            "update_rule": "Q[s,a] += alpha*(r + gamma*max Q[s',a'] - Q[s,a])",
            "states": f"{N_STATES} = 3 threat x 3 confidence x 2 criticality",
            "actions": config.RL_ACTIONS,
            "hyperparameters": {"episodes": config.RL_EPISODES,
                                "alpha": config.RL_ALPHA,
                                "gamma": config.RL_GAMMA,
                                "epsilon": "1.0 -> 0.05 (x0.9995/episode)"},
            "reward_design": REWARD,
            "baseline_comparison": {k: round(v, 3)
                                    for k, v in evals.items()},
            "how_it_learns": "The agent tries actions at random first; "
                             "every action returns a reward from the SOC "
                             "cost model. The Q-update strengthens state-"
                             "action pairs that earned high long-term "
                             "reward, so successful response strategies "
                             "emerge purely from reward feedback.",
        }
        save_json(os.path.join(config.REPORT_DIR, "rl_results.json"),
                  summary)
    return summary


if __name__ == "__main__":
    run()
