# -*- coding: utf-8 -*-
from pathlib import Path

path = Path('templates/index.html')
text = path.read_text(encoding='utf-8')
lines = text.splitlines()
start = 1618
end = 1991
new_block = '''    <!-- RIGHT: Audit Report -->
    <div class="report-panel">
        <div class="report-top-bar">
            <div class="report-title-group">
                <h2>Audit Report</h2>
            </div>
            <div style="display: flex; gap: 10px;" data-html2canvas-ignore="true">
                <button class="reset-btn" onclick="resetForm()">← Analyze Another</button>
            </div>
        </div>

        <div class="tab-nav" id="reportTabs">
            <button class="tab-btn active" onclick="switchTab('tab-overview', this)">Overview & Mitigation</button>
            <button class="tab-btn" onclick="switchTab('tab-charts', this)">Visualizations</button>
            <button class="tab-btn" onclick="switchTab('tab-compliance', this)">Legal & Compliance</button>
            <button class="tab-btn" onclick="switchTab('tab-human', this)">Human Impact</button>
            <button class="tab-btn" onclick="switchTab('tab-redteam', this)">Red Teaming</button>
            <button class="tab-btn" onclick="switchTab('tab-sandbox', this)">Simulation Lab</button>
            <button class="tab-btn" onclick="switchTab('tab-history', this)">Audit Log</button>
        </div>

        <div id="tab-overview" class="tab-pane active">
            <div class="grade-verdict-row">
                <div id="gradeCircle"></div>
                <div class="verdict-and-meta">
                    <div id="verdictBadge"></div>
                    <div id="metaRow" class="meta-row"></div>
                </div>
            </div>

            <div class="section-label">Fairness Metrics</div>
            <div class="metrics-grid" id="metricsGrid"></div>
            <div class="group-card" id="groupCard"></div>

            <div class="section-label">Gemini Analysis</div>
            <div class="explanation-card">
                <div class="explanation-header">
                    <div class="gemini-dot"></div>
                    <span>Gemini Analysis</span>
                </div>
                <div class="explanation-text" id="explanationText"></div>
            </div>

            <button class="mitigate-btn" id="mitigateBtn" onclick="generateStrategy()">
                ⚡ Generate Python Mitigation Script
            </button>
            <div class="strategy-loading" id="strategyLoading">
                <div style="margin-bottom: 12px;">Gemini is writing the python script...</div>
                <div class="progress-container" style="margin-top: 0; width: 100%;">
                    <div class="progress-fill" id="mitigateProgressFill"></div>
                </div>
                <div class="progress-percent" id="mitigateProgressPercent" style="margin-top: 8px;">0%</div>
            </div>
            <div class="strategy-card" id="strategyCard">
                <div class="strategy-header">
                    <div class="strategy-dot"></div>
                    <span>Auto-Generated Mitigation Script</span>
                </div>
                <div class="code-block-container">
                    <div class="code-header">
                        <span class="code-filename">mitigate_bias.py</span>
                        <div class="code-actions">
                            <button id="copyBtn" onclick="copyCode()">Copy</button>
                            <button onclick="downloadCode()">Download</button>
                        </div>
                    </div>
                    <pre class="code-body"><code id="strategyCode"></code></pre>
                </div>
            </div>
            <div class="strategy-card" id="fixedAssetsCard" style="display: none; background: rgba(52,211,153,0.05); border-color: rgba(52,211,153,0.3);">
                <div class="strategy-header">
                    <div class="strategy-dot" style="background: var(--good);"></div>
                    <span style="color: var(--good);">Download Mitigated Assets</span>
                </div>
                <div style="display: flex; gap: 10px; margin-top: 12px;">
                    <a href="/download-mitigated" class="reset-btn" style="border-color: var(--good); color: var(--good); flex: 1; text-align: center;">📥 Fixed Dataset</a>
                </div>
            </div>
        </div>

        <div id="tab-charts" class="tab-pane">
            <div class="shap-card" id="shapCard">
                <h3>Model Feature Importance (SHAP)</h3>
                <div class="shap-img-container">
                    <img id="shapImg" class="shap-img" alt="SHAP Plot">
                </div>
            </div>
            <div class="section-label">Default Charts</div>
            <div class="metrics-grid">
                <div class="metric-card" style="height: 280px;"><canvas id="chart1"></canvas></div>
                <div class="metric-card" style="height: 280px;"><canvas id="chart2"></canvas></div>
                <div class="metric-card" style="height: 280px;"><canvas id="chart3"></canvas></div>
                <div class="metric-card" style="height: 280px;"><canvas id="chart4"></canvas></div>
            </div>
        </div>

        <div id="tab-compliance" class="tab-pane">
            <div class="section-label">Regulatory Readiness</div>
            <div style="display: flex; flex-direction: column; gap: 16px;">
                <div class="card" style="padding: 20px;">
                    <h3 style="font-size: 14px; color: var(--text);">US EEOC Uniform Guidelines</h3>
                    <div id="badge-eeoc" style="margin-top: 10px;">Pending...</div>
                    <div id="detail-eeoc" style="font-size: 12px; color: var(--muted); margin-top: 10px;"></div>
                </div>
            </div>
        </div>

        <div id="tab-human" class="tab-pane">
            <div class="section-label" style="color: var(--good);">Human Impact Analysis</div>
            <button class="add-chart-btn" id="humanImpactBtn" onclick="generateHumanImpact()">👤 Analyze Human Impact</button>
            <div id="humanLoading" style="display: none; margin-top: 20px;">
                <div class="progress-container"><div class="progress-fill" id="humanProgressFill"></div></div>
                <div class="progress-percent" id="humanProgressPercent">0%</div>
            </div>
            <div id="humanResults" style="margin-top: 20px;"></div>
        </div>

        <div id="tab-redteam" class="tab-pane">
            <div class="section-label" style="color: var(--bad);">Offensive Red Teaming</div>
            <button class="add-chart-btn" id="redTeamBtn" onclick="runRedTeamAttack()">🔥 Launch Adversarial Attack</button>
            <div id="redTeamLoading" style="display: none; margin-top: 20px;">
                <div class="progress-container"><div class="progress-fill" id="redTeamProgressFill"></div></div>
                <div class="progress-percent" id="redTeamProgressPercent">0%</div>
            </div>
            <div id="redTeamResults" style="margin-top: 20px;"></div>
        </div>

        <div id="tab-sandbox" class="tab-pane">
            <div class="section-label">Noxis Simulation Lab</div>
            <div class="card" style="padding: 24px; background: var(--surface2); margin-bottom: 24px;">
                <div id="sandboxGrade"></div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
                    <div class="form-group">
                        <label id="label-priv">Privileged Group</label>
                        <input type="range" id="slider-priv" min="1" max="100" oninput="updateSandbox()">
                    </div>
                    <div class="form-group">
                        <label id="label-unpriv">Unprivileged Group</label>
                        <input type="range" id="slider-unpriv" min="1" max="100" oninput="updateSandbox()">
                    </div>
                </div>
            </div>
            <div class="card" style="padding: 24px;"><canvas id="ceoChart" style="height: 250px;"></canvas></div>
        </div>

        <div id="tab-history" class="tab-pane">
            <div class="section-label">Audit History</div>
            <div class="card" style="padding: 20px;">
                <div id="history-list"></div>
            </div>
        </div>
    </div>
</div>'''
lines[start:end] = new_block.splitlines()
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Replacement complete')
