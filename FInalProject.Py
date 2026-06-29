#!/usr/bin/env python3
"""
linear_gui_with_split.py
GUI trainer with manual train/test split (no sklearn split function).
Train/Test ratio default: test=0.34, train=0.66

Requirements:
    pip install numpy pandas matplotlib scikit-learn
(we use sklearn.metrics.r2_score only)
Run:
    python linear_gui_with_split.py
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import r2_score

# -------------------- Core ML functions --------------------

def load_csv(path, target_col, drop_cols=None):
    df = pd.read_csv(path)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in CSV columns")
    feature_names = df.columns.drop(target_col).tolist()
    X = df[feature_names].values.astype(float)
    y = df[target_col].values.astype(float)
    return X, y, feature_names

def standardize(X, eps=1e-8):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < eps] = 1.0
    Xs = (X - mu) / sigma
    return Xs, mu, sigma

def add_bias(X):
    n = X.shape[0]
    return np.hstack([np.ones((n,1)), X])

def predict(X_with_bias, W):
    return X_with_bias.dot(W)

def cost_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def manual_train_test_split(X, y, test_ratio=0.34, seed=0):

    np.random.seed(seed)
    n = X.shape[0]
    idx = np.random.permutation(n)
    test_size = int(np.round(n * test_ratio))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test

def train_batch_gd(X_with_bias, y, lr=0.1, epochs=200, l2_lambda=0.0, verbose=False, seed=0, progress_callback=None):
    np.random.seed(seed)
    n = X_with_bias.shape[0]
    d = X_with_bias.shape[1]
    W = np.random.randn(d) * 0.01
    loss_history = []
    for epoch in range(epochs):
        y_pred = predict(X_with_bias, W)
        loss = cost_mse(y, y_pred) + l2_lambda * np.sum(W[1:]**2)
        loss_history.append(loss)
        grad = (-2.0 / n) * X_with_bias.T.dot(y - y_pred)
        if l2_lambda > 0:
            reg = (2.0 * l2_lambda) * np.hstack([0.0, W[1:]])
            grad += reg
        W -= lr * grad
        if progress_callback and epoch % max(1, epochs//10) == 0:
            progress_callback(epoch, epochs, loss)
    return W, loss_history

def train_sgd(X_with_bias, y, lr=1e-3, epochs=100, batch_size=64,
                   l2_lambda=0.0, verbose=False, seed=0,
                   grad_clip_norm=5.0, backoff_factor=0.5, max_backoffs=5,
                   progress_callback=None):
    """
    SGD with safety checks:
    - default lr small (1e-3)
    - gradient clipping (L2 norm)
    - detect NaN/Inf and back off lr or stop
    """
    np.random.seed(seed)
    n = X_with_bias.shape[0]
    if n == 0:
        raise ValueError("Empty dataset")
    if not np.isfinite(X_with_bias).all() or not np.isfinite(y).all():
        raise ValueError("Input data contains NaN or Inf")

    d = X_with_bias.shape[1]
    W = np.random.randn(d) * 0.01
    loss_history = []
    backoffs = 0

    def clip_grad(grad, max_norm):
        norm = np.linalg.norm(grad)
        if norm > max_norm and norm > 0:
            return grad * (max_norm / norm)
        return grad

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X_sh = X_with_bias[perm]
        y_sh = y[perm]
        for i in range(0, n, batch_size):
            Xb = X_sh[i:i+batch_size]
            yb = y_sh[i:i+batch_size]
            nb = Xb.shape[0]
            if nb == 0:
                continue
            yb_pred = Xb.dot(W)
            grad = (-2.0 / nb) * Xb.T.dot(yb - yb_pred)
            if l2_lambda > 0:
                reg = (2.0 * l2_lambda) * np.hstack([0.0, W[1:]])
                grad += reg
            grad = clip_grad(grad, grad_clip_norm)
            W_new = W - lr * grad

            if not np.isfinite(grad).all():
                raise FloatingPointError(f"Found non-finite grad at epoch {epoch}, batch {i}")
            if not np.isfinite(W_new).all():
                backoffs += 1
                if backoffs <= max_backoffs:
                    lr *= backoff_factor
                    if verbose:
                        print(f"[WARN] Non-finite W_new detected. Reducing lr -> {lr:.2e} and retrying (backoff {backoffs})")
                    continue
                else:
                    raise FloatingPointError(f"Weights became non-finite after {backoffs} backoffs. Stopping.")
            W = W_new

        y_pred_full = X_with_bias.dot(W)
        loss = np.mean((y - y_pred_full)**2) + l2_lambda * np.sum(W[1:]**2)
        loss_history.append(loss)

        if progress_callback:
            progress_callback(epoch+1, epochs, loss)
        if verbose and (epoch % max(1, epochs//10) == 0):
            grad_norm = np.linalg.norm(grad)
            print(f"Epoch {epoch+1}/{epochs} - loss {loss:.6f} - grad_norm {grad_norm:.6f} - lr {lr:.2e}")

        if not np.isfinite(loss):
            raise FloatingPointError(f"Loss became non-finite at epoch {epoch}")

    return W, loss_history

def evaluate_model(X_with_bias, y, W):
    y_pred = predict(X_with_bias, W)
    mse = cost_mse(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # Accuracy definition for regression
    accuracy = r2 * 100.0

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy,
        "y_pred": y_pred
    }


# -------------------- GUI --------------------

class LinearTrainerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Linear Regression Trainer (manual train/test split)")
        self.geometry("1000x700")
        self.create_widgets()
        self.filepath = None
        self.feature_names = []
        self.loss_history = None
        self.train_metrics = None
        self.test_metrics = None
        self.W = None

    def create_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(frm, text="CSV file:").grid(row=0, column=0, sticky=tk.W)
        self.csv_entry = ttk.Entry(frm, width=60)
        self.csv_entry.grid(row=0, column=1, columnspan=3, padx=4)
        ttk.Button(frm, text="Browse", command=self.browse_file).grid(row=0, column=4, padx=4)

        ttk.Label(frm, text="Target column:").grid(row=1, column=0, sticky=tk.W)
        self.target_entry = ttk.Entry(frm, width=20)
        self.target_entry.insert(0, "target")
        self.target_entry.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(frm, text="Drop columns (comma sep):").grid(row=1, column=2, sticky=tk.W)
        self.drop_entry = ttk.Entry(frm, width=25)
        self.drop_entry.grid(row=1, column=3, sticky=tk.W)

        ttk.Label(frm, text="Optimizer:").grid(row=2, column=0, sticky=tk.W)
        self.optimizer_var = tk.StringVar(value="batch")
        ttk.Radiobutton(frm, text="Batch", variable=self.optimizer_var, value="batch").grid(row=2, column=1)
        ttk.Radiobutton(frm, text="SGD", variable=self.optimizer_var, value="sgd").grid(row=2, column=2)

        ttk.Label(frm, text="Learning rate:").grid(row=3, column=0, sticky=tk.W)
        self.lr_entry = ttk.Entry(frm, width=10); self.lr_entry.insert(0, "0.1")
        self.lr_entry.grid(row=3, column=1, sticky=tk.W)

        ttk.Label(frm, text="Epochs:").grid(row=3, column=2, sticky=tk.W)
        self.epochs_entry = ttk.Entry(frm, width=10); self.epochs_entry.insert(0, "300")
        self.epochs_entry.grid(row=3, column=3, sticky=tk.W)

        ttk.Label(frm, text="Batch size (SGD):").grid(row=3, column=4, sticky=tk.W)
        self.bs_entry = ttk.Entry(frm, width=8); self.bs_entry.insert(0, "32")
        self.bs_entry.grid(row=3, column=5, sticky=tk.W)

        ttk.Label(frm, text="Test size (0-1):").grid(row=3, column=6, sticky=tk.W)
        self.testsize_entry = ttk.Entry(frm, width=8); self.testsize_entry.insert(0, "0.34")
        self.testsize_entry.grid(row=3, column=7, sticky=tk.W)

        ttk.Label(frm, text="L2 lambda:").grid(row=4, column=0, sticky=tk.W)
        self.l2_entry = ttk.Entry(frm, width=10); self.l2_entry.insert(0, "0.0")
        self.l2_entry.grid(row=4, column=1, sticky=tk.W)

        ttk.Label(frm, text="Seed:").grid(row=4, column=2, sticky=tk.W)
        self.seed_entry = ttk.Entry(frm, width=10); self.seed_entry.insert(0, "0")
        self.seed_entry.grid(row=4, column=3, sticky=tk.W)

        self.verbose_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Verbose", variable=self.verbose_var).grid(row=4, column=4)

        # Y-axis min/max inputs
        ttk.Label(frm, text="Y-min:").grid(row=4, column=5, sticky=tk.W)
        self.ymin_entry = ttk.Entry(frm, width=10)
        self.ymin_entry.insert(0, "")  # empty = auto
        self.ymin_entry.grid(row=4, column=6, sticky=tk.W)

        ttk.Label(frm, text="Y-max:").grid(row=4, column=7, sticky=tk.W)
        self.ymax_entry = ttk.Entry(frm, width=10)
        self.ymax_entry.insert(0, "")  # empty = auto
        self.ymax_entry.grid(row=4, column=8, sticky=tk.W)

        # Buttons
        self.train_btn = ttk.Button(frm, text="Train", command=self.on_train)
        self.train_btn.grid(row=5, column=0, pady=8)
        ttk.Button(frm, text="Load features", command=self.load_features).grid(row=5, column=1)
        ttk.Button(frm, text="Reset plot", command=self.reset_plot).grid(row=5, column=2)
        ttk.Button(frm, text="Apply Y-range", command=self.apply_y_range).grid(row=5, column=3)

        # Metrics display
        metrics_frm = ttk.LabelFrame(self, text="Metrics")
        metrics_frm.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.metrics_text = tk.Text(metrics_frm, height=6, width=120)
        self.metrics_text.pack(side=tk.LEFT, padx=4, pady=4)

        # Plot area
        plot_frm = ttk.Frame(self)
        plot_frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.fig, self.ax = plt.subplots(figsize=(6,3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frm)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Progress label
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.progress_var).pack(side=tk.BOTTOM, anchor=tk.W, padx=8, pady=4)

    def browse_file(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if p:
            self.csv_entry.delete(0, tk.END)
            self.csv_entry.insert(0, p)
            self.filepath = p

    def load_features(self):
        path = self.csv_entry.get().strip()
        if not path:
            messagebox.showerror("Error", "Choose a CSV file first")
            return
        try:
            df = pd.read_csv(path, nrows=5)
            cols = list(df.columns)
            messagebox.showinfo("Columns", "Columns found:\n" + "\n".join(cols))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_plot(self):
        self.ax.clear()
        self.canvas.draw()

    def set_progress(self, epoch, total, loss):
        self.progress_var.set(f"Training... epoch {epoch}/{total}  loss={loss:.6f}")
        self.update_idletasks()

    def on_train(self):
        path = self.csv_entry.get().strip()
        if not path:
            messagebox.showerror("Error", "Choose a CSV file first")
            return
        target = self.target_entry.get().strip()
        drop_cols = [c.strip() for c in self.drop_entry.get().split(",") if c.strip()]
        optimizer = self.optimizer_var.get()
        try:
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.bs_entry.get())
            test_size = float(self.testsize_entry.get())
            if not (0.0 < test_size < 1.0):
                raise ValueError("test size must be between 0 and 1")
            l2 = float(self.l2_entry.get())
            seed = int(self.seed_entry.get())
            verbose = bool(self.verbose_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Invalid numeric input: {e}")
            return

        self.train_btn.config(state=tk.DISABLED)
        self.progress_var.set("Loading data...")

        def _train_thread():
            try:
                # load
                X, y, feature_names = load_csv(path, target, drop_cols if drop_cols else None)

                # manual split BEFORE standardization
                X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_ratio=test_size, seed=seed)

                # standardize using train stats, apply to test
                Xs_train, mu, sigma = standardize(X_train)
                Xs_test = (X_test - mu) / sigma

                Xb_train = add_bias(Xs_train)
                Xb_test = add_bias(Xs_test)

                self.feature_names = ["bias"] + feature_names

                # train on TRAIN only
                if optimizer == "batch":
                    W, loss_hist = train_batch_gd(Xb_train, y_train, lr=lr, epochs=epochs, l2_lambda=l2, verbose=verbose, seed=seed, progress_callback=self.set_progress if verbose else None)
                else:
                    W, loss_hist = train_sgd(Xb_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size, l2_lambda=l2, verbose=verbose, seed=seed, progress_callback=self.set_progress if verbose else None)

                # evaluate on both sets
                metrics_train = evaluate_model(Xb_train, y_train, W)
                metrics_test = evaluate_model(Xb_test, y_test, W)

                self.loss_history = loss_hist
                self.train_metrics = metrics_train
                self.test_metrics = metrics_test
                self.W = W

                self.after(0, lambda: self.on_training_done(W, metrics_train, metrics_test))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, lambda: self.train_btn.config(state=tk.NORMAL))

        t = threading.Thread(target=_train_thread, daemon=True)
        t.start()

    def apply_y_range(self):
        # redraw plot applying current Y-min/Y-max inputs (no re-training)
        if self.loss_history is None:
            messagebox.showinfo("Info", "No training history to plot yet.")
            return
        try:
            ymin_input = self.ymin_entry.get().strip()
            ymax_input = self.ymax_entry.get().strip()
            if ymin_input != "" and ymax_input != "":
                ymin = float(ymin_input)
                ymax = float(ymax_input)
                if ymin >= ymax:
                    raise ValueError("Y-min must be less than Y-max")
            else:
                ymin = min(self.loss_history)
                ymax = max(self.loss_history)
                pad = (ymax - ymin) * 0.2 if ymax > ymin else 1.0
                ymin = max(0, ymin - pad)
                ymax = ymax + pad
        except Exception as e:
            messagebox.showwarning("Warning", f"Invalid Y-range input, using auto-range. ({e})")
            ymin = min(self.loss_history)
            ymax = max(self.loss_history)
            pad = (ymax - ymin) * 0.2 if ymax > ymin else 1.0
            ymin = max(0, ymin - pad)
            ymax = ymax + pad

        self.ax.clear()
        self.ax.plot(self.loss_history)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_title('Training loss (MSE)')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('MSE')
        self.ax.grid(True)
        self.canvas.draw()

    def on_training_done(self, W, metrics_train, metrics_test):
        self.train_btn.config(state=tk.NORMAL)
        self.progress_var.set("Training finished")
        self.metrics_text.delete(1.0, tk.END)

        total_accuracy = (
            metrics_train["accuracy"] + metrics_test["accuracy"]
        ) / 2.0

        s = f"Learned params (bias then weights):\n{W}\n\n"
        s += f"Features: {self.feature_names}\n\n"

        s += "Metrics (TRAIN):\n"
        s += (
            f"  MSE={metrics_train['mse']:.6f}, "
            f"RMSE={metrics_train['rmse']:.6f}, "
            f"R2={metrics_train['r2']:.6f}, "
            f"Accuracy={metrics_train['accuracy']:.2f}%\n\n"
        )

        s += "Metrics (TEST):\n"
        s += (
            f"  MSE={metrics_test['mse']:.6f}, "
            f"RMSE={metrics_test['rmse']:.6f}, "
            f"R2={metrics_test['r2']:.6f}, "
            f"Accuracy={metrics_test['accuracy']:.2f}%\n\n"
        )

        s += f"TOTAL Accuracy (Train + Test avg): {total_accuracy:.2f}%\n"

        self.metrics_text.insert(tk.END, s)

        # Plot loss
        self.ax.clear()
        if self.loss_history is not None:
            self.ax.plot(self.loss_history)

            ymin_input = self.ymin_entry.get().strip()
            ymax_input = self.ymax_entry.get().strip()

            try:
                if ymin_input != "" and ymax_input != "":
                    ymin = float(ymin_input)
                    ymax = float(ymax_input)
                    if ymin >= ymax:
                        raise ValueError
                else:
                    ymin = min(self.loss_history)
                    ymax = max(self.loss_history)
                    pad = (ymax - ymin) * 0.2 if ymax > ymin else 1.0
                    ymin = max(0, ymin - pad)
                    ymax = ymax + pad
            except:
                ymin = min(self.loss_history)
                ymax = max(self.loss_history)
                pad = (ymax - ymin) * 0.2 if ymax > ymin else 1.0
                ymin = max(0, ymin - pad)
                ymax = ymax + pad

            self.ax.set_ylim(ymin, ymax)
            self.ax.set_title("Training loss (MSE)")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("MSE")
            self.ax.grid(True)

        self.canvas.draw()


if __name__ == '__main__':
    app = LinearTrainerGUI()
    app.mainloop()
