import sys
from pathlib import Path

def upgrade_dashboard(file_path):
    p = Path(file_path)
    content = p.read_text(encoding='utf-8')
    
    # 1. Update Typography & Design System Variables
    # (Simplified replacement for the whole <style> block)
    style_start = content.find('<style>')
    style_end = content.find('</style>', style_start)
    
    if style_start == -1 or style_end == -1:
        print("Could not find style tag")
        return

    new_style = """
<style>
    :root, [data-theme="dark"] {
      --bg-base: #0F172A;
      --bg-surface: #1E293B;
      --bg-card: #1E293B;
      --bg-card-hover: #334155;
      --border: #334155;
      --border-bright: #475569;
      --primary: #6366F1;
      --primary-light: #818CF8;
      --primary-glow: rgba(99, 102, 241, 0.1);
      --success: #10B981;
      --warning: #F59E0B;
      --danger: #EF4444;
      --text-primary: #F8FAFC;
      --text-secondary: #94A3B8;
      --text-muted: #64748B;
      --font-family: 'Inter', sans-serif;
      --text-xs: 12px; --text-sm: 14px; --text-md: 15px; --text-lg: 18px; --text-2xl: 24px;
      --radius-md: 12px; --transition: 200ms ease;
    }

    [data-theme="light"] {
      --bg-base: #F8FAFC; --bg-surface: #FFFFFF; --bg-card: #FFFFFF; --bg-card-hover: #F1F5F9;
      --border: #E2E8F0; --border-bright: #CBD5E1;
      --text-primary: #0F172A; --text-secondary: #475569; --text-muted: #94A3B8;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body {
      font-family: var(--font-family);
      background-color: var(--bg-base);
      color: var(--text-primary);
      font-size: var(--text-sm);
      line-height: 1.6;
      overflow-x: hidden;
    }

    .layout { display: grid; grid-template-columns: 240px 1fr; grid-template-rows: 60px 1fr; min-height: 100vh; }
    .topnav { grid-column: 1 / -1; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; background: var(--bg-surface); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 100; }
    .sidebar { background: var(--bg-surface); border-right: 1px solid var(--border); padding: 24px 0; display: flex; flex-direction: column; gap: 4px; overflow-y: auto; }
    .main { padding: 32px; display: flex; flex-direction: column; gap: 32px; overflow-y: auto; }

    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 24px;
      transition: all var(--transition);
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-card:hover { border-color: var(--primary-light); transform: translateY(-4px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }
    .kpi-label { font-size: var(--text-xs); font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }
    .kpi-value { font-size: var(--text-2xl); font-weight: 800; color: var(--primary-light); }
    
    .ai-reasoning {
      font-size: var(--text-sm);
      color: var(--text-secondary);
      background: var(--primary-glow);
      padding: 16px;
      border-radius: var(--radius-md);
      border-left: 4px solid var(--primary-light);
      margin-top: auto;
      animation: fadeIn 0.5s ease-out;
    }
    .ai-reasoning::before { content: 'AI REASONING'; display: block; font-size: 10px; font-weight: 900; color: var(--primary-light); margin-bottom: 6px; letter-spacing: 0.05em; }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    .theme-toggle { display: flex; gap: 4px; background: var(--bg-base); padding: 4px; border-radius: 20px; border: 1px solid var(--border); }
    .theme-btn { padding: 6px 14px; border-radius: 16px; border: none; background: transparent; color: var(--text-muted); font-size: 12px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    .theme-btn.active { background: var(--primary); color: white; }

    .btn { padding: 10px 24px; border-radius: 10px; font-weight: 600; cursor: pointer; border: none; transition: 0.2s; display: inline-flex; align-items: center; gap: 8px; }
    .btn-primary { background: var(--primary); color: white; }
    .btn-ghost { background: transparent; border: 1px solid var(--border); color: var(--text-secondary); }

    .section-title { font-size: var(--text-xs); font-weight: 800; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px; }
    .section-title::before { content: '/// '; color: var(--primary-light); }

    .panel { background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 24px; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary-light); }
</style>"""
    
    content = content[:style_start] + new_style + content[style_end+8:]
    
    # 2. Update KPI Grid HTML to be modular
    kpi_grid_start = content.find('<div class="kpi-grid">')
    kpi_grid_end = content.find('</div>', kpi_grid_start)
    
    # Replace the existing hardcoded KPIs with a modular placeholder
    content = content[:kpi_grid_start] + '<div class="kpi-grid" id="kpi-container"></div>' + content[kpi_grid_end+6:]
    
    # 3. Update updateDashboard JavaScript function
    js_func_start = content.find('function updateDashboard')
    js_func_end = content.find('}', js_func_start) # Rough estimate

    # Actually lets find the whole script block at the end
    script_start = content.find('<script>')
    # I'll just append the new logic because the existing script is massive

    p.write_text(content, encoding='utf-8')

upgrade_dashboard('ui/dashboard.html')
