"""
Terminal UI for Live SMC Signal Monitor
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os
import json
import pyperclip

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich import box
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
import threading
import queue
import time
try:
    import keyboard
except ImportError:
    keyboard = None
from plyer import notification

from .live_smc_engine import LiveSMCEngine

class LiveMonitorTUI:
    """Terminal User Interface for Live SMC Monitor"""
    
    def __init__(self, symbol: str, config: Dict):
        self.symbol = symbol.upper()
        self.config = config
        self.console = Console()
        
        # State
        self.signals = []  # All historical signals
        self.signal_history_file = f"signals_history_{self.symbol}.json"
        self.current_price = 0.0
        self.htf_bias = 'neutral'
        self.status = 'starting'
        self.stats = {
            'total_signals': 0,
            'accuracy': 0.0,
            'avg_rr': 0.0
        }
        
        # Copy functionality
        self.selected_signal_index = 0  # For copying specific signal data
        self.last_copy_message = ""
        
        # Signal status checking
        self.status_check_interval = config.get('status_check_interval', 45)  # seconds
        self.last_status_check = 0
        
        # Threading
        self.signal_queue = queue.Queue()
        self.update_queue = queue.Queue()
        self.running = True
        
        # SMC Engine
        self.engine = LiveSMCEngine(symbol, config)
        self.engine.add_signal_callback(self._on_new_signal)
        self.engine.add_update_callback(self._on_data_update)
        
        # Load signal history
        self._load_signal_history()
        
    async def start(self):
        """Start the live monitor"""
        self.console.clear()
        
        # Start keyboard listener in separate thread
        keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        # Redirect logging during TUI operation to avoid console spam
        import logging
        logging.getLogger().handlers = [h for h in logging.getLogger().handlers if not isinstance(h, logging.StreamHandler)]
        
        # Start SMC engine
        engine_task = asyncio.create_task(self.engine.start())
        
        # Start UI
        try:
            with Live(self._create_layout(), refresh_per_second=0.2, console=self.console, screen=False) as live:
                self.live = live
                self.status = 'running'
                self.last_update = 0
                
                while self.running:
                    # Process signal queue
                    signals_changed = self._process_signal_queue()
                    
                    # Process update queue  
                    data_changed = self._process_update_queue()
                    
                    # Check signal statuses periodically
                    current_time = time.time()
                    status_changed = False
                    if (current_time - self.last_status_check) > self.status_check_interval:
                        status_changed = self._update_signal_statuses()
                        self.last_status_check = current_time
                    
                    # Update display only if something changed or every 10 seconds
                    if signals_changed or data_changed or status_changed or (current_time - self.last_update) > 10:
                        live.update(self._create_layout())
                        self.last_update = current_time
                    
                    await asyncio.sleep(2.0)
                    
        except KeyboardInterrupt:
            pass
        finally:
            await self.engine.stop()
            self.status = 'stopped'
            
    def _create_layout(self) -> Layout:
        """Create the main layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Header
        layout["header"].update(self._create_header())
        
        # Main content
        layout["left"].update(self._create_signals_panel())
        layout["right"].update(self._create_info_panel())
        
        # Footer
        layout["footer"].update(self._create_footer())
        
        return layout
        
    def _create_header(self) -> Panel:
        """Create header panel"""
        status_color = {
            'starting': 'yellow',
            'running': 'green',
            'stopped': 'red'
        }.get(self.status, 'white')
        
        bias_color = {
            'bull': 'green',
            'bear': 'red',
            'neutral': 'yellow'
        }.get(self.htf_bias, 'white')
        
        header_text = Text()
        header_text.append("🎯 SMC Live Signal Monitor v1.0", style="bold cyan")
        header_text.append(f" | {self.symbol}", style="bold white")
        header_text.append(f" | Status: ", style="white")
        header_text.append(f"{self.status.upper()}", style=f"bold {status_color}")
        header_text.append(f" | Price: ${self.current_price:.2f}", style="bold white")
        header_text.append(f" | HTF Bias: ", style="white")
        header_text.append(f"{self.htf_bias.upper()}", style=f"bold {bias_color}")
        
        # WebSocket status
        engine_status = self.engine.get_status()
        ws_connected = engine_status.get('websocket_connected', False)
        ws_status = "🟢 WS" if ws_connected else "🔴 WS"
        ws_color = "green" if ws_connected else "red"
        header_text.append(f" | ", style="white")
        header_text.append(ws_status, style=f"bold {ws_color}")
        
        header_text.append(f" | {datetime.now().strftime('%H:%M:%S')}", style="dim white")
        
        return Panel(Align.center(header_text), box=box.ROUNDED)
        
    def _create_signals_panel(self) -> Panel:
        """Create signals table panel"""
        table = Table(title="🚨 Live Signals", box=box.ROUNDED)
        
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Dir", style="bold", width=5)
        table.add_column("Entry", style="yellow", width=10)
        table.add_column("SL", style="red", width=10)
        table.add_column("TP", style="green", width=10)
        table.add_column("RR", style="magenta", width=6)
        table.add_column("Conf", style="blue", width=6)
        table.add_column("Status", style="white", width=8)
        
        # Show last 15 signals
        recent_signals = self.signals[-15:] if len(self.signals) > 15 else self.signals
        
        for i, signal in enumerate(recent_signals):
            time_str = signal['timestamp'].strftime('%H:%M')
            direction = signal['direction']
            dir_style = "bold green" if direction == "LONG" else "bold red"
            
            # Determine status based on relevance, not time
            status_info = self._calculate_signal_status(signal)
            status = status_info['display']
                
            table.add_row(
                time_str,
                Text(direction, style=dir_style),
                f"${signal['entry']:.2f}",
                f"${signal['sl']:.2f}",
                f"${signal['tp']:.2f}",
                f"{signal['rr']:.1f}",
                signal.get('confidence', 'med'),
                status
            )
            
        if not self.signals:
            table.add_row("-", "-", "-", "-", "-", "-", "-", "[dim]Waiting for signals...[/]")
            
        return Panel(table, title="Signals", border_style="blue")
        
    def _create_info_panel(self) -> Panel:
        """Create info panel with stats and controls"""
        # Statistics
        stats_table = Table(title="📊 Statistics", box=box.SIMPLE)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        # Calculate status counts
        status_counts = {'valid': 0, 'triggered': 0, 'hit_tp': 0, 'hit_sl': 0, 'missed': 0}
        for signal in self.signals:
            status_info = self._calculate_signal_status(signal)
            status = status_info['status']
            if status in status_counts:
                status_counts[status] += 1
        
        stats_table.add_row("Total Signals", str(len(self.signals)))
        stats_table.add_row("Valid", f"[green]{status_counts['valid']}[/]")
        stats_table.add_row("Triggered", f"[cyan]{status_counts['triggered']}[/]") 
        stats_table.add_row("Hit TP", f"[bold green]{status_counts['hit_tp']}[/]")
        stats_table.add_row("Hit SL", f"[bold red]{status_counts['hit_sl']}[/]")
        stats_table.add_row("Missed", f"[orange1]{status_counts['missed']}[/]")
        stats_table.add_row("HTF Bias", self.htf_bias.title())
        stats_table.add_row("Avg RR", f"{self.stats['avg_rr']:.1f}")
        
        # Win rate calculation
        total_closed = status_counts['hit_tp'] + status_counts['hit_sl']
        if total_closed > 0:
            win_rate = (status_counts['hit_tp'] / total_closed) * 100
            stats_table.add_row("Win Rate", f"{win_rate:.1f}%")
        
        # WebSocket status
        engine_status = self.engine.get_status()
        ws_connected = engine_status.get('websocket_connected', False)
        ws_status = "🟢 Connected" if ws_connected else "🔴 Disconnected"
        stats_table.add_row("WebSocket", ws_status)
        
        # Controls
        controls_text = Text("\n🎛️ Controls:\n", style="bold cyan")
        if keyboard:
            controls_text.append("[T] Test Signal\n", style="white")
            controls_text.append("\n📋 Copy (latest signal):\n", style="bold yellow")
            controls_text.append("[C] Entry Price\n", style="white")
            controls_text.append("[L] SL Price\n", style="white") 
            controls_text.append("[V] TP Price\n", style="white")
            controls_text.append("[A] All Info\n", style="white")
            controls_text.append("[P] Market Price\n", style="white")
            controls_text.append("\n[Del] Clear Signals\n", style="white")
            controls_text.append("[Q] Quit\n", style="white")
            controls_text.append(f"\n⏰ Status check: {self.status_check_interval}s", style="dim cyan")
            
            # Show last copy message
            if self.last_copy_message:
                controls_text.append(f"\n💬 {self.last_copy_message}", style="green")
                
            controls_text.append("\nKeyboard: ✅ Active", style="green")
        else:
            controls_text.append("Use Ctrl+C to quit\n", style="white")
            controls_text.append("Keyboard: ❌ Disabled", style="red")
        
        content = Columns([stats_table, controls_text], equal=False)
        
        return Panel(content, title="Info & Controls", border_style="green")
        
    def _create_footer(self) -> Panel:
        """Create footer panel"""
        engine_status = self.engine.get_status()
        
        footer_text = Text()
        footer_text.append("📡 Data: ", style="white")
        footer_text.append(f"LTF: {engine_status.get('ltf_candles', 0)} candles", style="green")
        footer_text.append(" | ", style="dim white")
        footer_text.append(f"HTF: {engine_status.get('htf_candles', 0)} candles", style="green")
        footer_text.append(" | ", style="dim white")
        footer_text.append("Last Update: ", style="white")
        last_update = engine_status.get('last_analysis')
        if last_update:
            footer_text.append(last_update.strftime('%H:%M:%S'), style="cyan")
        else:
            footer_text.append("Never", style="dim red")
            
        return Panel(Align.center(footer_text), box=box.ROUNDED)
        
    async def _on_new_signal(self, signal: Dict):
        """Handle new signal from engine"""
        self.signal_queue.put(('signal', signal))
        
    async def _on_data_update(self, timeframe: str, data: Dict):
        """Handle data update from engine"""
        self.update_queue.put(('update', {'timeframe': timeframe, 'data': data}))
        
    def _process_signal_queue(self) -> bool:
        """Process signal queue"""
        changed = False
        try:
            while True:
                msg_type, data = self.signal_queue.get_nowait()
                if msg_type == 'signal':
                    # Check for duplicate signals (within 1 minute)
                    is_duplicate = False
                    signal_time = data.get('timestamp', datetime.now())
                    for existing_signal in self.signals:
                        time_diff = abs((existing_signal['timestamp'] - signal_time).total_seconds())
                        if (time_diff < 60 and 
                            existing_signal['direction'] == data['direction'] and
                            abs(existing_signal['entry'] - data['entry']) < 0.01):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        self.signals.append(data)
                        self._save_signal_to_history(data)
                        self._show_signal_notification(data)
                        self._update_stats()
                        changed = True
        except queue.Empty:
            pass
        return changed
            
    def _process_update_queue(self) -> bool:
        """Process update queue"""
        changed = False
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                if msg_type == 'update':
                    # Update current state
                    engine_status = self.engine.get_status()
                    new_price = engine_status.get('current_price', 0.0) or 0.0
                    new_bias = engine_status.get('htf_bias', 'neutral')
                    
                    # Check if anything actually changed
                    if (abs(new_price - self.current_price) > 0.01 or 
                        new_bias != self.htf_bias):
                        changed = True
                        
                    self.current_price = new_price
                    self.htf_bias = new_bias
        except queue.Empty:
            pass
        return changed
            
    def _show_signal_notification(self, signal: Dict):
        """Show desktop notification for new signal"""
        try:
            direction_emoji = "📈" if signal['direction'] == 'LONG' else "📉"
            
            notification.notify(
                title=f"🚨 SMC Signal - {self.symbol}",
                message=f"{direction_emoji} {signal['direction']} @ ${signal['entry']:.2f}\n"
                       f"SL: ${signal['sl']:.2f} | TP: ${signal['tp']:.2f}\n"
                       f"RR: {signal['rr']:.1f} | Confidence: {signal.get('confidence', 'medium')}",
                timeout=10
            )
        except Exception as e:
            # Notification might fail on some systems
            pass
            
    def _update_stats(self):
        """Update statistics"""
        if self.signals:
            total_rr = sum(s.get('rr', 0) for s in self.signals)
            self.stats['avg_rr'] = total_rr / len(self.signals)
            
    def _keyboard_listener(self):
        """Listen for keyboard input"""
        if not keyboard:
            # Fallback: just wait and check self.running periodically
            while self.running:
                time.sleep(0.5)
            return
            
        try:
            while self.running:
                if keyboard.is_pressed('q'):
                    self.running = False
                    break
                elif keyboard.is_pressed('t'):
                    self._test_signal()
                elif keyboard.is_pressed('c'):
                    self._copy_entry_price()
                elif keyboard.is_pressed('l'):
                    self._copy_sl_price()
                elif keyboard.is_pressed('v'):
                    self._copy_tp_price()
                elif keyboard.is_pressed('a'):
                    self._copy_all_signal_info()
                elif keyboard.is_pressed('p'):
                    self._copy_market_price()
                elif keyboard.is_pressed('delete'):
                    self.signals.clear()
                    
                time.sleep(0.1)
        except Exception as e:
            # If keyboard library fails, just wait
            while self.running:
                time.sleep(0.5)
            
    def _test_signal(self):
        """Add test signal for demonstration"""
        test_signal = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'direction': 'LONG',
            'entry': self.current_price or 2650.00,
            'sl': (self.current_price or 2650.00) * 0.98,
            'tp': (self.current_price or 2650.00) * 1.06,
            'rr': 3.0,
            'htf_bias': 'bull',
            'fvg_confluence': True,
            'confidence': 'high'
        }
        self.signals.append(test_signal)
        
    def _load_signal_history(self):
        """Load signal history from file"""
        try:
            if os.path.exists(self.signal_history_file):
                with open(self.signal_history_file, 'r') as f:
                    data = json.load(f)
                    # Convert timestamp strings back to datetime
                    for signal in data:
                        signal['timestamp'] = datetime.fromisoformat(signal['timestamp'])
                    self.signals = data
                    # Keep only last 100 signals to avoid memory issues
                    if len(self.signals) > 100:
                        self.signals = self.signals[-100:]
                    self._update_stats()
        except Exception as e:
            # If loading fails, start with empty history
            self.signals = []
            
    def _save_signal_to_history(self, signal: Dict):
        """Save single signal to history file"""
        try:
            # Convert datetime to string for JSON serialization
            signal_copy = signal.copy()
            signal_copy['timestamp'] = signal_copy['timestamp'].isoformat()
            
            # Load existing signals
            existing_signals = []
            if os.path.exists(self.signal_history_file):
                try:
                    with open(self.signal_history_file, 'r') as f:
                        existing_signals = json.load(f)
                except:
                    existing_signals = []
            
            # Add new signal
            existing_signals.append(signal_copy)
            
            # Keep only last 100 signals
            if len(existing_signals) > 100:
                existing_signals = existing_signals[-100:]
                
            # Save back to file
            with open(self.signal_history_file, 'w') as f:
                json.dump(existing_signals, f, indent=2)
        except Exception:
            pass  # Silently handle file errors
            
    def _copy_entry_price(self):
        """Copy entry price of latest signal to clipboard"""
        if not self.signals:
            self.last_copy_message = "No signals to copy"
            return
            
        latest_signal = self.signals[-1]
        entry_price = f"{latest_signal['entry']:.2f}"
        
        try:
            pyperclip.copy(entry_price)
            self.last_copy_message = f"Copied entry: ${entry_price}"
        except Exception:
            self.last_copy_message = "Copy failed - no clipboard"
            
    def _copy_sl_price(self):
        """Copy SL price of latest signal to clipboard"""
        if not self.signals:
            self.last_copy_message = "No signals to copy"
            return
            
        latest_signal = self.signals[-1]
        sl_price = f"{latest_signal['sl']:.2f}"
        
        try:
            pyperclip.copy(sl_price)
            self.last_copy_message = f"Copied SL: ${sl_price}"
        except Exception:
            self.last_copy_message = "Copy failed - no clipboard"
            
    def _copy_tp_price(self):
        """Copy TP price of latest signal to clipboard"""
        if not self.signals:
            self.last_copy_message = "No signals to copy"
            return
            
        latest_signal = self.signals[-1]
        tp_price = f"{latest_signal['tp']:.2f}"
        
        try:
            pyperclip.copy(tp_price)
            self.last_copy_message = f"Copied TP: ${tp_price}"
        except Exception:
            self.last_copy_message = "Copy failed - no clipboard"
            
    def _copy_all_signal_info(self):
        """Copy all signal information to clipboard"""
        if not self.signals:
            self.last_copy_message = "No signals to copy"
            return
            
        latest_signal = self.signals[-1]
        signal_text = (
            f"Signal: {latest_signal['direction']} {self.symbol}\n"
            f"Entry: ${latest_signal['entry']:.2f}\n"
            f"SL: ${latest_signal['sl']:.2f}\n"
            f"TP: ${latest_signal['tp']:.2f}\n"
            f"RR: {latest_signal['rr']:.1f}\n"
            f"Time: {latest_signal['timestamp'].strftime('%H:%M:%S')}"
        )
        
        try:
            pyperclip.copy(signal_text)
            self.last_copy_message = "Copied all signal info"
        except Exception:
            self.last_copy_message = "Copy failed - no clipboard"
            
    def _copy_market_price(self):
        """Copy current market price to clipboard"""
        if not self.current_price:
            self.last_copy_message = "No market price available"
            return
            
        price_text = f"{self.current_price:.2f}"
        
        try:
            pyperclip.copy(price_text)
            self.last_copy_message = f"Copied price: ${price_text}"
        except Exception:
            self.last_copy_message = "Copy failed - no clipboard"
            
    def _calculate_signal_status(self, signal: Dict) -> Dict:
        """Calculate signal status based on market conditions"""
        current_price = self.current_price or 0
        if not current_price:
            return {'status': 'unknown', 'display': '[dim]WAITING[/]'}
            
        entry = signal['entry']
        sl = signal['sl']
        tp = signal['tp']
        direction = signal['direction']
        signal_time = signal['timestamp']
        age_minutes = (datetime.now() - signal_time).total_seconds() / 60
        
        # NEW: First 5 minutes
        if age_minutes < 5:
            return {'status': 'new', 'display': '[bold yellow]● NEW[/]'}
            
        # HIT_TP: Price reached take profit
        if direction == 'LONG' and current_price >= tp:
            return {'status': 'hit_tp', 'display': '[bold green]✅ HIT TP[/]'}
        elif direction == 'SHORT' and current_price <= tp:
            return {'status': 'hit_tp', 'display': '[bold green]✅ HIT TP[/]'}
            
        # HIT_SL: Price hit stop loss
        if direction == 'LONG' and current_price <= sl:
            return {'status': 'hit_sl', 'display': '[bold red]❌ HIT SL[/]'}
        elif direction == 'SHORT' and current_price >= sl:
            return {'status': 'hit_sl', 'display': '[bold red]❌ HIT SL[/]'}
            
        # TRIGGERED: Price reached entry (within 0.1% tolerance)
        entry_tolerance = entry * 0.001
        if abs(current_price - entry) <= entry_tolerance:
            return {'status': 'triggered', 'display': '[bold cyan]⚡ TRIGGERED[/]'}
            
        # MISSED: Price moved past entry without hitting (by >2%)
        if direction == 'LONG':
            if current_price > entry * 1.02:  # Price went too high past entry
                return {'status': 'missed', 'display': '[bold orange]⏭️ MISSED[/]'}
        else:  # SHORT
            if current_price < entry * 0.98:  # Price went too low past entry
                return {'status': 'missed', 'display': '[bold orange]⏭️ MISSED[/]'}
                
        # VALID: Price still between SL and TP, entry not yet reached
        if direction == 'LONG':
            if sl < current_price < entry:
                return {'status': 'valid', 'display': '[bold white]📈 VALID[/]'}
        else:  # SHORT
            if entry < current_price < sl:
                return {'status': 'valid', 'display': '[bold white]📉 VALID[/]'}
                
        # EXPIRED: Too old (>2 hours) and not triggered
        if age_minutes > 120:
            return {'status': 'expired', 'display': '[dim]⏰ EXPIRED[/]'}
            
        # DEFAULT: Active but unclear
        return {'status': 'active', 'display': '[dim]⏳ ACTIVE[/]'}
        
    def _update_signal_statuses(self) -> bool:
        """Update all signal statuses and return if any changed"""
        if not self.signals:
            return False
            
        changed = False
        for signal in self.signals:
            old_status = signal.get('calculated_status', {}).get('status', '')
            new_status_info = self._calculate_signal_status(signal)
            
            if old_status != new_status_info['status']:
                signal['calculated_status'] = new_status_info
                changed = True
                
        return changed
        
async def run_live_monitor(symbol: str, config: Dict):
    """Run the live monitor"""
    monitor = LiveMonitorTUI(symbol, config)
    await monitor.start()
