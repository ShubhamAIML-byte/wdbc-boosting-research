/**
 * WDBC Reliability-Aware Boosting Inference Engine
 * Client-side inference for AdaBoost, XGBoost, CatBoost, and SLSQP-Optimized Ensemble
 */

(function (window) {
  'use strict';

  const Engine = {};

  // Sigmoid activation
  function sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-25, Math.min(25, x))));
  }

  // Standardize raw features using WDBC training mean & scale
  Engine.standardize = function (rawFeatures) {
    const data = window.WDBC_DATA;
    const mean = data.scaler.mean;
    const scale = data.scaler.scale;
    const std = [];
    for (let i = 0; i < rawFeatures.length; i++) {
      std.push((rawFeatures[i] - mean[i]) / scale[i]);
    }
    return std;
  };

  // Unstandardize back to physical cytology units
  Engine.unstandardize = function (stdFeatures) {
    const data = window.WDBC_DATA;
    const mean = data.scaler.mean;
    const scale = data.scaler.scale;
    const raw = [];
    for (let i = 0; i < stdFeatures.length; i++) {
      raw.push(stdFeatures[i] * scale[i] + mean[i]);
    }
    return raw;
  };

  // 1. AdaBoost Evaluation (100 DecisionTreeClassifier trees of depth 2)
  Engine.predictAdaBoost = function (stdFeatures) {
    const data = window.WDBC_DATA;
    if (!data.ada_trees) return 0.5;

    let score = 0;
    const trees = data.ada_trees;

    for (let t = 0; t < trees.length; t++) {
      const tree = trees[t];
      const left = tree.children_left;
      const right = tree.children_right;
      const feat = tree.feature;
      const thresh = tree.threshold;
      const val = tree.value;
      const weight = tree.weight;

      let node = 0;
      while (left[node] !== -1 && right[node] !== -1) {
        const fIdx = feat[node];
        if (stdFeatures[fIdx] <= thresh[node]) {
          node = left[node];
        } else {
          node = right[node];
        }
      }

      // Leaf node: val[node] is [count_0, count_1]
      const c0 = val[node][0] || 0;
      const c1 = val[node][1] || 0;
      const vote = c1 > c0 ? 1 : -1;
      score += weight * vote;
    }

    // AdaBoost probability via logistic sigmoid of margin
    return sigmoid(score * 0.15); // scaled margin
  };

  // 2. XGBoost Evaluation (200 trees of max depth 3)
  Engine.predictXGBoost = function (stdFeatures) {
    const data = window.WDBC_DATA;
    if (!data.xgb_trees_json) return 0.5;

    let margin = 0;
    const trees = data.xgb_trees_json;

    for (let t = 0; t < trees.length; t++) {
      let node = trees[t];
      while (node.children && node.children.length > 0) {
        const splitFeat = parseInt(node.split.replace('f', ''), 10);
        const splitVal = node.split_condition;
        const featureVal = stdFeatures[splitFeat];

        if (featureVal < splitVal) {
          node = node.children[0].nodeid === node.yes ? node.children[0] : node.children[1];
        } else {
          node = node.children[0].nodeid === node.no ? node.children[0] : node.children[1];
        }
      }
      if (node.leaf !== undefined) {
        margin += node.leaf;
      }
    }

    return sigmoid(margin);
  };

  // 3. CatBoost Evaluation (200 oblivious symmetric trees of depth 5)
  Engine.predictCatBoost = function (stdFeatures) {
    const data = window.WDBC_DATA;
    if (!data.cat_model || !data.cat_model.trees) return 0.5;

    let margin = 0;
    const trees = data.cat_model.trees;

    for (let t = 0; t < trees.length; t++) {
      const tree = trees[t];
      const splits = tree.splits;
      let leafIdx = 0;

      for (let s = 0; s < splits.length; s++) {
        const split = splits[s];
        if (stdFeatures[split.feature] > split.border) {
          leafIdx |= (1 << s);
        }
      }

      margin += tree.leaf_values[leafIdx] || 0;
    }

    // Apply scale and bias if present
    const sb = data.cat_model.scale_and_bias;
    if (sb && sb.length > 0 && sb[0].length >= 2) {
      margin = sb[0][0] * margin + sb[0][1];
    }

    return sigmoid(margin);
  };

  // 4. Surrogate Logistic Regression (for gradient computation & FGSM attack)
  Engine.predictSurrogate = function (stdFeatures) {
    const data = window.WDBC_DATA;
    const coef = data.surrogate.coefficients;
    const intercept = data.surrogate.intercept;

    let z = intercept;
    for (let i = 0; i < stdFeatures.length; i++) {
      z += coef[i] * stdFeatures[i];
    }
    return sigmoid(z);
  };

  // 5. FGSM Adversarial Perturbation Generator
  // Transfer-based attack: grad = (p_surrogate - y_true) * coef
  Engine.applyFGSMAttack = function (stdFeatures, yTrue, epsilon) {
    if (epsilon <= 0) {
      return {
        attackedStd: [...stdFeatures],
        attackedRaw: Engine.unstandardize(stdFeatures),
        deltas: new Array(stdFeatures.length).fill(0),
        gradients: new Array(stdFeatures.length).fill(0)
      };
    }

    const data = window.WDBC_DATA;
    const coef = data.surrogate.coefficients;
    const pSurrogate = Engine.predictSurrogate(stdFeatures);
    const pClipped = Math.max(1e-7, Math.min(1 - 1e-7, pSurrogate));

    const attackedStd = [];
    const deltas = [];
    const gradients = [];

    for (let i = 0; i < stdFeatures.length; i++) {
      const grad = (pClipped - yTrue) * coef[i];
      gradients.push(grad);
      const sign = grad > 0 ? 1 : grad < 0 ? -1 : 0;
      const delta = epsilon * sign;
      deltas.push(delta);
      attackedStd.push(stdFeatures[i] + delta);
    }

    const attackedRaw = Engine.unstandardize(attackedStd);

    return {
      attackedStd,
      attackedRaw,
      deltas,
      gradients
    };
  };

  // 6. Full Ensemble Predictor
  Engine.predict = function (rawFeatures, options = {}) {
    const data = window.WDBC_DATA;
    const epsilon = options.epsilon || 0.0;
    const yPseudo = options.yPseudo !== undefined ? options.yPseudo : 1; // 1 = malignant target attack direction

    // Standardize
    let std = Engine.standardize(rawFeatures);
    let attackInfo = null;

    // Apply FGSM attack if epsilon > 0
    if (epsilon > 0) {
      attackInfo = Engine.applyFGSMAttack(std, yPseudo, epsilon);
      std = attackInfo.attackedStd;
    }

    // Individual Model Predictions
    const pAda = Engine.predictAdaBoost(std);
    const pXgb = Engine.predictXGBoost(std);
    const pCat = Engine.predictCatBoost(std);

    // Weights from OOF SLSQP
    const weights = data.ensemble_weights;
    const wXgb = weights.XGBoost;
    const wAda = weights.AdaBoost;
    const wCat = weights.CatBoost;

    // Weighted Ensemble Probability
    const pOptimized = pXgb * wXgb + pAda * wAda + pCat * wCat;
    const pEqual = (pXgb + pAda + pCat) / 3.0;

    // Classification & Confidence
    const isMalignant = pOptimized >= 0.5;
    const diagnosis = isMalignant ? 'Malignant' : 'Benign';
    const confidence = isMalignant ? pOptimized : (1.0 - pOptimized);

    // Clinical Risk Tier
    let riskTier = 'Low Risk';
    let riskColor = '#10b981';
    if (pOptimized >= 0.85) {
      riskTier = 'High Risk (Critical)';
      riskColor = '#f43f5e';
    } else if (pOptimized >= 0.5) {
      riskTier = 'Elevated Risk (Suspect)';
      riskColor = '#fb923c';
    } else if (pOptimized >= 0.25) {
      riskTier = 'Borderline (Follow-up Recommended)';
      riskColor = '#facc15';
    }

    // Local SHAP Feature Attribution Approximation
    const localAttributions = Engine.calculateLocalAttributions(std, pOptimized);

    return {
      rawFeatures: options.rawFeatures || rawFeatures,
      stdFeatures: std,
      attackInfo,
      epsilon,
      probabilities: {
        XGBoost: pXgb,
        AdaBoost: pAda,
        CatBoost: pCat,
        equalVoting: pEqual,
        optimizedEnsemble: pOptimized
      },
      weights: {
        XGBoost: wXgb,
        AdaBoost: wAda,
        CatBoost: wCat
      },
      diagnosis,
      isMalignant,
      probability: pOptimized,
      confidence,
      riskTier,
      riskColor,
      localAttributions
    };
  };

  // 7. Calculate Local Patient Feature Attribution (SHAP-aligned)
  Engine.calculateLocalAttributions = function (stdFeatures, probMalignant) {
    const data = window.WDBC_DATA;
    const shapFeatures = data.shap_features;
    const attributions = [];

    // Map feature name to global importance and correlation sign
    const featMap = {};
    for (let f of shapFeatures) {
      featMap[f.feature] = f;
    }

    for (let i = 0; i < data.feature_names.length; i++) {
      const fName = data.feature_names[i];
      const zScore = stdFeatures[i];
      const meta = data.feature_metadata[fName];
      const globalInfo = featMap[fName] || { importance: 0.1 };

      // In oncology cytology, higher nuclear dimension/irregularity correlates positively with malignancy
      // z-score > 0 indicates cell nucleus is larger/more irregular than average
      const baseDirection = (meta.malignant_mean > meta.benign_mean) ? 1.0 : -1.0;
      const contribution = zScore * globalInfo.importance * baseDirection;

      attributions.push({
        feature: fName,
        index: i,
        zScore,
        rawValue: meta.mean + zScore * meta.scale,
        category: meta.category,
        globalImportance: globalInfo.importance,
        contribution: contribution,
        pushesMalignant: contribution > 0
      });
    }

    // Sort by absolute contribution magnitude
    attributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    return attributions;
  };

  window.WDBCEngine = Engine;

})(typeof window !== 'undefined' ? window : this);
