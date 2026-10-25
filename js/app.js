/**
 * WDBC ResilientBoost Application Controller
 * Handles UI interactions, state, real-time recalculations, charts, and reporting
 */

(function () {
  'use strict';

  // Global State
  const state = {
    data: window.WDBC_DATA,
    engine: window.WDBCEngine,
    currentPatient: null,
    pristineFeatures: {},
    currentEpsilon: 0.0,
    currentCategory: 'all',
    cohortFilter: 'all',
    searchQuery: '',
    theme: 'dark'
  };

  // DOM Elements
  const DOM = {};

  function initDOM() {
    DOM.themeToggle = document.getElementById('theme-toggle-btn');
    DOM.themeIcon = document.getElementById('theme-icon');
    DOM.navTabs = document.querySelectorAll('.nav-tab-btn');
    DOM.tabPanels = document.querySelectorAll('.tab-panel');

    // Hero Elements
    DOM.cancerGauge = document.getElementById('cancer-gauge-fill');
    DOM.gaugeRiskPct = document.getElementById('gauge-risk-pct');
    DOM.verdictBadge = document.getElementById('hero-verdict-badge');
    DOM.verdictText = document.getElementById('hero-verdict-text');
    DOM.riskTier = document.getElementById('hero-risk-tier');
    DOM.patientId = document.getElementById('hero-patient-id');
    DOM.verdictDesc = document.getElementById('hero-verdict-desc');

    // Tri-Model Bars
    DOM.catProb = document.getElementById('cat-prob-display');
    DOM.catBar = document.getElementById('cat-bar-fill');
    DOM.xgbProb = document.getElementById('xgb-prob-display');
    DOM.xgbBar = document.getElementById('xgb-bar-fill');
    DOM.adaProb = document.getElementById('ada-prob-display');
    DOM.adaBar = document.getElementById('ada-bar-fill');

    // Presets
    DOM.presetBtns = document.querySelectorAll('.preset-chip[data-preset]');
    DOM.randomPresetBtn = document.getElementById('preset-random-btn');

    // Lab Sliders
    DOM.categoryFilterBar = document.getElementById('category-filter-bar');
    DOM.sliderGrid = document.getElementById('feature-slider-grid');
    DOM.resetFeaturesBtn = document.getElementById('reset-features-btn');
    DOM.applyNoiseBtn = document.getElementById('apply-noise-btn');

    // Cohort
    DOM.cohortSearch = document.getElementById('cohort-search-input');
    DOM.cohortContainer = document.getElementById('cohort-list-container');
    DOM.cohortCounter = document.getElementById('cohort-counter');
    DOM.filterAllCohort = document.getElementById('filter-all-cohort');
    DOM.filterBenignCohort = document.getElementById('filter-benign-cohort');
    DOM.filterMaligCohort = document.getElementById('filter-malig-cohort');

    // Adversarial Lab
    DOM.epsilonSlider = document.getElementById('epsilon-slider');
    DOM.epsilonValDisplay = document.getElementById('epsilon-val-display');
    DOM.epsPresetBtns = document.querySelectorAll('.eps-preset-btn');
    DOM.attackStatusTag = document.getElementById('attack-status-tag');
    DOM.robustnessTableBody = document.getElementById('robustness-table-body');
    DOM.attackDeltasContainer = document.getElementById('patient-attack-deltas-container');

    // SHAP Studio
    DOM.waterfallContainer = document.getElementById('patient-waterfall-container');
    DOM.globalShapList = document.getElementById('global-shap-list');

    // Benchmarks
    DOM.cleanBenchmarkTableBody = document.getElementById('clean-benchmark-table-body');

    // Report
    DOM.printReportBtn = document.getElementById('print-report-btn');
    DOM.downloadJsonBtn = document.getElementById('download-report-json-btn');
    DOM.repPatientId = document.getElementById('rep-patient-id');
    DOM.repDate = document.getElementById('rep-date');
    DOM.repVerdict = document.getElementById('rep-verdict');
    DOM.repConfidence = document.getElementById('rep-confidence');
    DOM.repRiskTier = document.getElementById('rep-risk-tier');
    DOM.repCatProb = document.getElementById('rep-cat-prob');
    DOM.repXgbProb = document.getElementById('rep-xgb-prob');
    DOM.repAdaProb = document.getElementById('rep-ada-prob');
    DOM.repShapTopDrivers = document.getElementById('rep-shap-top-drivers');
  }

  // Switch Active Tab
  window.switchTab = function (tabId) {
    DOM.navTabs.forEach(btn => {
      if (btn.getAttribute('data-tab') === tabId) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    DOM.tabPanels.forEach(panel => {
      if (panel.id === tabId) {
        panel.classList.add('active');
      } else {
        panel.classList.remove('active');
      }
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Toggle Dark/Light Theme
  function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('wdbc_theme', state.theme);
    updateThemeIcon();
  }

  function updateThemeIcon() {
    if (state.theme === 'light') {
      DOM.themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"></path>';
    } else {
      DOM.themeIcon.innerHTML = '<path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>';
    }
  }

  // Load Preset Patient
  function loadPreset(index) {
    const preset = state.data.patient_cases[index];
    if (!preset) return;

    state.currentPatient = {
      id: preset.id,
      title: preset.title,
      description: preset.description,
      true_diagnosis: preset.true_diagnosis,
      true_label: preset.true_label,
      features: { ...preset.features }
    };
    state.pristineFeatures = { ...preset.features };

    // Update preset buttons state
    DOM.presetBtns.forEach(btn => {
      if (parseInt(btn.getAttribute('data-preset'), 10) === index) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    renderAll();
  }

  // Load Patient from Cohort by Index
  function loadCohortPatient(cohortIndex) {
    const pt = state.data.test_cohort[cohortIndex];
    if (!pt) return;

    state.currentPatient = {
      id: pt.id,
      title: `Case: ${pt.id} (${pt.true_diagnosis})`,
      description: `Test partition patient #${cohortIndex}. Ground truth label: ${pt.true_diagnosis}. Evaluated across all 3 boosting classifiers.`,
      true_diagnosis: pt.true_diagnosis,
      true_label: pt.true_label,
      features: { ...pt.features }
    };
    state.pristineFeatures = { ...pt.features };

    // Deselect preset buttons
    DOM.presetBtns.forEach(btn => btn.classList.remove('active'));

    renderAll();
  }

  // Load Random Patient
  function loadRandomPatient() {
    const cohort = state.data.test_cohort;
    const randomIdx = Math.floor(Math.random() * cohort.length);
    loadCohortPatient(randomIdx);
  }

  // Recalculate and Update Predictions
  function updateInference() {
    if (!state.currentPatient) return;

    const rawArray = state.data.feature_names.map(f => state.currentPatient.features[f]);
    const yPseudo = state.currentPatient.true_label === 1 ? 0 : 1; // attack direction tries to flip truth

    const result = state.engine.predict(rawArray, {
      epsilon: state.currentEpsilon,
      yPseudo: yPseudo
    });

    state.lastInference = result;

    // Update Hero Verdict & Gauge
    const probPct = (result.probability * 100).toFixed(1);
    DOM.gaugeRiskPct.textContent = `${probPct}%`;

    // Radial Gauge stroke calculation (circumference = 2 * PI * 70 ≈ 440)
    const circumference = 440;
    const offset = circumference - (result.probability * circumference);
    DOM.cancerGauge.style.strokeDashoffset = Math.max(0, Math.min(circumference, offset));

    if (result.isMalignant) {
      DOM.cancerGauge.style.stroke = 'var(--malignant)';
      DOM.verdictBadge.className = 'verdict-badge malignant';
      DOM.verdictText.textContent = 'MALIGNANT (NEOPLASTIC)';
    } else {
      DOM.cancerGauge.style.stroke = 'var(--benign)';
      DOM.verdictBadge.className = 'verdict-badge benign';
      DOM.verdictText.textContent = 'BENIGN (LOW RISK)';
    }

    DOM.riskTier.textContent = `Stratification: ${result.riskTier}`;
    DOM.riskTier.style.color = result.riskColor;
    DOM.patientId.textContent = `ID: ${state.currentPatient.id}`;
    DOM.verdictDesc.textContent = state.currentPatient.description;

    // Update Tri-Model Bars
    const catProbVal = (result.probabilities.CatBoost * 100).toFixed(2);
    const xgbProbVal = (result.probabilities.XGBoost * 100).toFixed(2);
    const adaProbVal = (result.probabilities.AdaBoost * 100).toFixed(2);

    DOM.catProb.textContent = `${catProbVal}%`;
    DOM.xgbProb.textContent = `${xgbProbVal}%`;
    DOM.adaProb.textContent = `${adaProbVal}%`;

    DOM.catBar.style.width = `${Math.min(100, Math.max(2, catProbVal))}%`;
    DOM.xgbBar.style.width = `${Math.min(100, Math.max(2, xgbProbVal))}%`;
    DOM.adaBar.style.width = `${Math.min(100, Math.max(2, adaProbVal))}%`;

    DOM.catBar.style.background = result.probabilities.CatBoost >= 0.5 ? 'var(--malignant)' : 'var(--benign)';
    DOM.xgbBar.style.background = result.probabilities.XGBoost >= 0.5 ? 'var(--malignant)' : 'var(--primary)';
    DOM.adaBar.style.background = result.probabilities.AdaBoost >= 0.5 ? 'var(--malignant)' : 'var(--purple)';

    // Update Dynamic Waterfall
    renderWaterfall(result.localAttributions);

    // Update Attack Deltas
    renderAttackDeltas(result);

    // Update Clinical Report
    updateReport(result);
  }

  // Render Feature Sliders Grid
  function renderSliders() {
    DOM.sliderGrid.innerHTML = '';
    const fNames = state.data.feature_names;
    const meta = state.data.feature_metadata;

    fNames.forEach(f => {
      const fMeta = meta[f];
      if (state.currentCategory !== 'all' && fMeta.category !== state.currentCategory) {
        return;
      }

      const val = state.currentPatient.features[f];
      const min = Math.max(0, fMeta.min * 0.5);
      const max = fMeta.max * 1.5;
      const step = (max - min) / 200;

      const item = document.createElement('div');
      item.className = 'slider-item';
      item.id = `slider-item-${fMeta.index}`;

      item.innerHTML = `
        <div class="slider-label-row">
          <span class="slider-name" title="${fMeta.description || ''}">${f}</span>
          <span class="slider-val-badge" id="slider-val-${fMeta.index}">${val.toFixed(3)} ${fMeta.unit || ''}</span>
        </div>
        <div class="slider-control-row">
          <input type="range" 
                 id="input-range-${fMeta.index}" 
                 min="${min}" 
                 max="${max}" 
                 step="${step}" 
                 value="${val}">
        </div>
        <div class="slider-benchmarks">
          <span class="b-norm" title="Dataset Benign Average">Benign Avg: ${fMeta.benign_mean.toFixed(2)}</span>
          <span class="m-norm" title="Dataset Malignant Average">Malig Avg: ${fMeta.malignant_mean.toFixed(2)}</span>
        </div>
      `;

      const input = item.querySelector('input');
      input.addEventListener('input', (e) => {
        const newVal = parseFloat(e.target.value);
        state.currentPatient.features[f] = newVal;
        document.getElementById(`slider-val-${fMeta.index}`).textContent = `${newVal.toFixed(3)} ${fMeta.unit || ''}`;
        updateInference();
      });

      DOM.sliderGrid.appendChild(item);
    });
  }

  // Update slider positions when patient changes
  function updateSliderValues() {
    state.data.feature_names.forEach((f, idx) => {
      const val = state.currentPatient.features[f];
      const input = document.getElementById(`input-range-${idx}`);
      const valBadge = document.getElementById(`slider-val-${idx}`);
      const meta = state.data.feature_metadata[f];
      if (input) input.value = val;
      if (valBadge) valBadge.textContent = `${val.toFixed(3)} ${meta.unit || ''}`;
    });
  }

  // Render Cohort List
  function renderCohort() {
    DOM.cohortContainer.innerHTML = '';
    const cohort = state.data.test_cohort;
    let count = 0;

    cohort.forEach((pt, i) => {
      // Filtering
      if (state.cohortFilter === 'benign' && pt.true_label !== 0) return;
      if (state.cohortFilter === 'malignant' && pt.true_label !== 1) return;
      if (state.searchQuery) {
        const query = state.searchQuery.toLowerCase();
        const matchId = pt.id.toLowerCase().includes(query);
        const matchDiag = pt.true_diagnosis.toLowerCase().includes(query);
        if (!matchId && !matchDiag) return;
      }

      count++;
      const item = document.createElement('div');
      item.className = 'cohort-patient-item';
      if (state.currentPatient && state.currentPatient.id === pt.id) {
        item.classList.add('selected');
      }

      const labelClass = pt.true_label === 0 ? 'benign' : 'malignant';
      const probPercent = (pt.prob_ensemble * 100).toFixed(1);

      item.innerHTML = `
        <div>
          <span class="patient-id-badge">${pt.id}</span>
          <span style="font-size:0.75rem; color:var(--text-dim); margin-left:0.4rem;">Risk: ${probPercent}%</span>
        </div>
        <span class="patient-label-tag ${labelClass}">${pt.true_diagnosis}</span>
      `;

      item.addEventListener('click', () => {
        loadCohortPatient(i);
        document.querySelectorAll('.cohort-patient-item').forEach(el => el.classList.remove('selected'));
        item.classList.add('selected');
      });

      DOM.cohortContainer.appendChild(item);
    });

    DOM.cohortCounter.textContent = `${count} cases shown`;
  }

  // Render Waterfall Chart
  function renderWaterfall(attributions) {
    DOM.waterfallContainer.innerHTML = '';
    // Show top 10 features
    const top10 = attributions.slice(0, 10);
    const maxVal = Math.max(...top10.map(a => Math.abs(a.contribution)), 0.01);

    top10.forEach(attr => {
      const row = document.createElement('div');
      row.className = 'waterfall-row';

      const pct = Math.min(100, (Math.abs(attr.contribution) / maxVal) * 100);
      const isMalig = attr.pushesMalignant;
      const fillClass = isMalig ? 'malignant' : 'benign';
      const sign = isMalig ? '+' : '-';

      row.innerHTML = `
        <span class="waterfall-feat-name" title="${attr.feature} (${attr.category})">${attr.feature}</span>
        <div class="waterfall-bar-track">
          <div class="waterfall-bar-fill ${fillClass}" style="width: ${pct}%;"></div>
        </div>
        <span class="waterfall-val-label" style="color: ${isMalig ? 'var(--malignant-light)' : 'var(--benign-light)'}">
          ${sign}${Math.abs(attr.contribution).toFixed(3)}
        </span>
      `;

      DOM.waterfallContainer.appendChild(row);
    });
  }

  // Render Active Perturbation Deltas
  function renderAttackDeltas(result) {
    DOM.attackDeltasContainer.innerHTML = '';
    if (!result.attackInfo || state.currentEpsilon === 0) {
      DOM.attackDeltasContainer.innerHTML = `
        <div style="font-size:0.85rem; color:var(--text-dim); text-align:center; padding:1.5rem 0;">
          No adversarial perturbation applied (&epsilon; = 0.00). Features are pristine.
        </div>
      `;
      DOM.attackStatusTag.textContent = 'Clean Baseline';
      DOM.attackStatusTag.className = 'patient-label-tag benign';
      return;
    }

    DOM.attackStatusTag.textContent = `Perturbation &epsilon; = ${state.currentEpsilon.toFixed(2)}`;
    DOM.attackStatusTag.className = 'patient-label-tag malignant';

    const deltas = result.attackInfo.deltas;
    const fNames = state.data.feature_names;
    const meta = state.data.feature_metadata;

    // Pick top 8 features with largest absolute delta
    const deltaItems = [];
    for (let i = 0; i < deltas.length; i++) {
      if (Math.abs(deltas[i]) > 0.0001) {
        deltaItems.push({
          name: fNames[i],
          deltaStd: deltas[i],
          origRaw: state.currentPatient.features[fNames[i]],
          attackedRaw: result.attackInfo.attackedRaw[i],
          unit: meta[fNames[i]].unit
        });
      }
    }

    deltaItems.sort((a, b) => Math.abs(b.deltaStd) - Math.abs(a.deltaStd));

    deltaItems.slice(0, 8).forEach(item => {
      const card = document.createElement('div');
      card.style.background = 'rgba(0,0,0,0.2)';
      card.style.padding = '0.5rem 0.75rem';
      card.style.borderRadius = 'var(--radius-sm)';
      card.style.display = 'flex';
      card.style.justifyContent = 'space-between';
      card.style.alignItems = 'center';
      card.style.fontSize = '0.8rem';

      const shiftPct = ((item.attackedRaw - item.origRaw) / (item.origRaw || 1) * 100).toFixed(1);
      const isPositive = item.deltaStd > 0;

      card.innerHTML = `
        <div>
          <span style="font-weight:600; text-transform:capitalize;">${item.name}</span>
          <div style="font-size:0.7rem; color:var(--text-dim);">
            ${item.origRaw.toFixed(2)} &rarr; ${item.attackedRaw.toFixed(2)} ${item.unit}
          </div>
        </div>
        <span style="font-family:var(--font-mono); font-weight:700; color:${isPositive ? 'var(--malignant-light)' : 'var(--primary-light)'}">
          ${isPositive ? '+' : ''}${shiftPct}%
        </span>
      `;

      DOM.attackDeltasContainer.appendChild(card);
    });
  }

  // Render Degradation Table
  function renderRobustnessTable() {
    DOM.robustnessTableBody.innerHTML = '';
    const tests = state.data.robustness_test;

    tests.forEach(row => {
      const tr = document.createElement('tr');
      const isSelected = Math.abs(row.epsilon - state.currentEpsilon) < 0.01;
      if (isSelected) tr.style.background = 'rgba(6, 182, 212, 0.12)';

      const accPct = (row.accuracy * 100).toFixed(2);
      const precPct = (row.precision * 100).toFixed(2);
      const sensPct = (row.sensitivity * 100).toFixed(2);
      const specPct = (row.specificity * 100).toFixed(2);
      const f1Pct = (row.f1 * 100).toFixed(2);
      const aucVal = row.roc_auc.toFixed(4);
      const dropBadgeClass = row.accuracy_drop_pp > 5 ? 'drop-badge-bad' : 'drop-badge-good';

      tr.innerHTML = `
        <td><strong>&epsilon; = ${row.epsilon.toFixed(2)}</strong></td>
        <td>${accPct}%</td>
        <td>${precPct}%</td>
        <td>${sensPct}%</td>
        <td>${specPct}%</td>
        <td>${f1Pct}%</td>
        <td>${aucVal}</td>
        <td class="${dropBadgeClass}">-${row.accuracy_drop_pp.toFixed(2)} pp</td>
      `;

      DOM.robustnessTableBody.appendChild(tr);
    });
  }

  // Render Global SHAP List
  function renderGlobalSHAP() {
    DOM.globalShapList.innerHTML = '';
    const list = state.data.shap_features.slice(0, 12);
    const maxImp = list[0].importance;

    list.forEach((item, i) => {
      const el = document.createElement('div');
      el.style.display = 'flex';
      el.style.flexDirection = 'column';
      el.style.gap = '0.2rem';
      el.style.fontSize = '0.8rem';

      const pct = (item.importance / maxImp) * 100;

      el.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
          <span style="font-weight:600; text-transform:capitalize;">${i + 1}. ${item.feature}</span>
          <span style="font-family:var(--font-mono); color:var(--primary-light); font-weight:600;">${item.importance.toFixed(3)}</span>
        </div>
        <div style="height:4px; background:rgba(255,255,255,0.08); border-radius:2px; overflow:hidden;">
          <div style="height:100%; width:${pct}%; background:linear-gradient(90deg, var(--primary), var(--purple)); border-radius:2px;"></div>
        </div>
      `;

      DOM.globalShapList.appendChild(el);
    });
  }

  // Render Clean Benchmark Table
  function renderBenchmarkTable() {
    DOM.cleanBenchmarkTableBody.innerHTML = '';
    const metrics = state.data.clean_metrics;

    metrics.forEach(m => {
      const tr = document.createElement('tr');
      const isOptimized = m.model.includes('Optimized');
      if (isOptimized) {
        tr.style.background = 'rgba(6, 182, 212, 0.12)';
        tr.style.fontWeight = '700';
      }

      tr.innerHTML = `
        <td><strong>${m.model}</strong></td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${(m.precision * 100).toFixed(2)}%</td>
        <td>${(m.sensitivity * 100).toFixed(2)}%</td>
        <td>${(m.specificity * 100).toFixed(2)}%</td>
        <td>${(m.f1 * 100).toFixed(2)}%</td>
        <td>${m.roc_auc.toFixed(4)}</td>
      `;

      DOM.cleanBenchmarkTableBody.appendChild(tr);
    });
  }

  // Update Pathology Report Content
  function updateReport(result) {
    if (!DOM.repPatientId) return;

    DOM.repPatientId.textContent = state.currentPatient.id;
    DOM.repDate.textContent = new Date().toISOString().split('T')[0];

    if (result.isMalignant) {
      DOM.repVerdict.textContent = 'MALIGNANT NEOPLASM';
      DOM.repVerdict.className = 'report-verdict-text malignant';
    } else {
      DOM.repVerdict.textContent = 'BENIGN CYTOPATHOLOGY';
      DOM.repVerdict.className = 'report-verdict-text benign';
    }

    const confPct = (result.confidence * 100).toFixed(1);
    const riskPct = (result.probability * 100).toFixed(2);
    DOM.repConfidence.textContent = `Ensemble Confidence: ${confPct}% | Malignancy Risk: ${riskPct}%`;
    DOM.repRiskTier.textContent = `Risk Stratification: ${result.riskTier}`;

    DOM.repCatProb.textContent = `${(result.probabilities.CatBoost * 100).toFixed(2)}%`;
    DOM.repXgbProb.textContent = `${(result.probabilities.XGBoost * 100).toFixed(2)}%`;
    DOM.repAdaProb.textContent = `${(result.probabilities.AdaBoost * 100).toFixed(2)}%`;

    // Top 3 SHAP drivers in report
    DOM.repShapTopDrivers.innerHTML = '';
    result.localAttributions.slice(0, 3).forEach(attr => {
      const p = document.createElement('div');
      p.style.display = 'flex';
      p.style.justifyContent = 'space-between';
      p.style.padding = '0.35rem 0.5rem';
      p.style.background = '#f1f5f9';
      p.style.borderRadius = '4px';

      const dir = attr.pushesMalignant ? 'Elevates Malignancy' : 'Supports Benign Finding';
      const color = attr.pushesMalignant ? '#e11d48' : '#059669';

      p.innerHTML = `
        <span style="font-weight:600; text-transform:capitalize;">${attr.feature} (Val: ${attr.rawValue.toFixed(2)})</span>
        <span style="color:${color}; font-weight:600;">${dir} (${attr.contribution > 0 ? '+' : ''}${attr.contribution.toFixed(3)})</span>
      `;
      DOM.repShapTopDrivers.appendChild(p);
    });
  }

  // Full Initial Render
  function renderAll() {
    renderSliders();
    renderCohort();
    renderRobustnessTable();
    renderGlobalSHAP();
    renderBenchmarkTable();
    updateInference();
  }

  // Setup Event Listeners
  function setupEventListeners() {
    // Theme Toggle
    DOM.themeToggle.addEventListener('click', toggleTheme);

    // Navigation Tabs
    DOM.navTabs.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tabId = btn.getAttribute('data-tab');
        window.switchTab(tabId);
      });
    });

    // Category Filter Bar
    DOM.categoryFilterBar.addEventListener('click', (e) => {
      const btn = e.target.closest('.category-filter-btn');
      if (!btn) return;
      document.querySelectorAll('.category-filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentCategory = btn.getAttribute('data-cat');
      renderSliders();
    });

    // Preset Buttons
    DOM.presetBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.getAttribute('data-preset'), 10);
        loadPreset(idx);
      });
    });

    // Random Preset Button
    DOM.randomPresetBtn.addEventListener('click', loadRandomPatient);

    // Reset Patient Features
    DOM.resetFeaturesBtn.addEventListener('click', () => {
      state.currentPatient.features = { ...state.pristineFeatures };
      state.currentEpsilon = 0.0;
      DOM.epsilonSlider.value = 0.0;
      DOM.epsilonValDisplay.textContent = '0.00';
      DOM.epsPresetBtns.forEach(b => {
        if (b.getAttribute('data-eps') === '0.00') b.classList.add('active');
        else b.classList.remove('active');
      });
      updateSliderValues();
      updateInference();
    });

    // Apply Cytology Noise
    DOM.applyNoiseBtn.addEventListener('click', () => {
      state.data.feature_names.forEach(f => {
        const delta = (Math.random() - 0.5) * 0.10 * state.currentPatient.features[f];
        state.currentPatient.features[f] = Math.max(0, state.currentPatient.features[f] + delta);
      });
      updateSliderValues();
      updateInference();
    });

    // Cohort Search
    DOM.cohortSearch.addEventListener('input', (e) => {
      state.searchQuery = e.target.value.trim();
      renderCohort();
    });

    // Cohort Filters
    DOM.filterAllCohort.addEventListener('click', () => {
      state.cohortFilter = 'all';
      renderCohort();
    });
    DOM.filterBenignCohort.addEventListener('click', () => {
      state.cohortFilter = 'benign';
      renderCohort();
    });
    DOM.filterMaligCohort.addEventListener('click', () => {
      state.cohortFilter = 'malignant';
      renderCohort();
    });

    // Epsilon Slider
    DOM.epsilonSlider.addEventListener('input', (e) => {
      const eps = parseFloat(e.target.value);
      state.currentEpsilon = eps;
      DOM.epsilonValDisplay.textContent = eps.toFixed(2);
      DOM.epsPresetBtns.forEach(b => {
        if (Math.abs(parseFloat(b.getAttribute('data-eps')) - eps) < 0.01) {
          b.classList.add('active');
        } else {
          b.classList.remove('active');
        }
      });
      updateInference();
      renderRobustnessTable();
    });

    // Epsilon Preset Buttons
    DOM.epsPresetBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const eps = parseFloat(btn.getAttribute('data-eps'));
        state.currentEpsilon = eps;
        DOM.epsilonSlider.value = eps;
        DOM.epsilonValDisplay.textContent = eps.toFixed(2);
        DOM.epsPresetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateInference();
        renderRobustnessTable();
      });
    });

    // Print Report
    DOM.printReportBtn.addEventListener('click', () => {
      window.print();
    });

    // Download JSON Report
    DOM.downloadJsonBtn.addEventListener('click', () => {
      if (!state.lastInference) return;
      const exportPayload = {
        patient_id: state.currentPatient.id,
        timestamp: new Date().toISOString(),
        diagnosis: state.lastInference.diagnosis,
        malignancy_probability: state.lastInference.probability,
        confidence: state.lastInference.confidence,
        risk_tier: state.lastInference.riskTier,
        epsilon_perturbation: state.currentEpsilon,
        model_probabilities: state.lastInference.probabilities,
        top_shap_features: state.lastInference.localAttributions.slice(0, 10),
        cytology_features: state.currentPatient.features
      };

      const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `WDBC_Report_${state.currentPatient.id}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // Application Startup
  function init() {
    initDOM();

    // Check saved theme
    const savedTheme = localStorage.getItem('wdbc_theme');
    if (savedTheme) {
      state.theme = savedTheme;
      document.documentElement.setAttribute('data-theme', savedTheme);
    }
    updateThemeIcon();

    // Load Default Preset (Case 1: Benign Fibroadenoma)
    loadPreset(0);

    // Setup Listeners
    setupEventListeners();
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
