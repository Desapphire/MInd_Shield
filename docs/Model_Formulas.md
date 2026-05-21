# Mind-Shield+ Model Formulas and Math Reference

This document collects the key formulas used across the project for paper-ready reference.

## 1) Cognitive Load (Weighted Fusion)

$$
\text{Load} = 0.25 \cdot S_{typing} + 0.25 \cdot S_{error} + 0.20 \cdot S_{attention} + 0.15 \cdot S_{idle} + 0.15 \cdot S_{engagement}
$$

Each component score is normalized to $[0, 100]$.

## 2) Risk Score (Unified Evaluation)

$$
\text{Risk} = 0.30 \cdot \text{Load} + 0.40 \cdot (100 \cdot P_{fatigue}) + 0.30 \cdot (100 \cdot D_{drift})
$$

Where $P_{fatigue}$ is the fatigue probability and $D_{drift}$ is the drift magnitude normalized to $[0,1]$.

## 3) Fatigue Prediction (Logistic Form)

A logistic model can be represented as:

$$
P_{fatigue} = \sigma(\beta^T x) = \frac{1}{1 + e^{-\beta^T x}}
$$

The project also supports Random Forest and Gradient Boosting models for improved non-linear performance.

## 4) Behavioral Drift (Isolation Forest)

Isolation Forest uses the average path length $E(h(x))$ in random trees. A shorter path indicates stronger anomaly:

$$
\text{AnomalyScore}(x) = 2^{-\frac{E(h(x))}{c(n)}}
$$

Where $c(n)$ is the average path length of unsuccessful searches in a binary tree of size $n$.

## 5) Face Authentication (Cosine Similarity)

$$
\text{sim}(a,b) = \frac{a \cdot b}{\|a\|\,\|b\|}
$$

If $\text{sim}(a,b)$ exceeds the threshold, the face is accepted.

## 6) Exponential Moving Average (Smoothing)

$$
S_t = \alpha x_t + (1 - \alpha) S_{t-1}
$$

Used to smooth posture and other noisy signals. Typical $\alpha \in [0.3, 0.5]$.

## 7) Posture Angles (Simplified)

Example head tilt from ear heights:

$$
\theta_{tilt} = \arctan\left(\frac{y_{left\_ear} - y_{right\_ear}}{x_{left\_ear} - x_{right\_ear}}\right)
$$

## 8) Typing Speed (WPM Approximation)

$$
\text{WPM} = \frac{\text{keystrokes} / 5}{\text{minutes}}
$$

## 9) Error Rate

$$
\text{ErrorRate} = \frac{\text{backspaces}}{\text{total keystrokes}}
$$
