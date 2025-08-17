"""
Terminal UI for Live SMC Signal Monitor
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

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
        self.signals = []
        self.current_price = 0.0
        self.htf_bias = 'neutral'
        self.status = 'starting'
        self.stats = {
            'total_signals': 0,
            'accuracy': 0.0,
            'avg_rr': 0.0
        }
        
        # Threading
        self.signal_queue = queue.Queue()
        self.update_queue = queue.Queue()
        self.running = True
        
        # SMC Engine
        self.engine = LiveSMCEngine(symbol, config)
        self.engine.add_signal_callback(self._on_new_signal)
        self.engine.add_update_callback(self._on_data_update)
        
    async def start(self):
        """Start the live monitor"""
        self.console.clear()
        
        # Start keyboard listener in separate thread
        keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        # Start SMC engine
        engine_task = asyncio.create_task(self.engine.start())
        
        # Start UI
        try:
            with Live(self._create_layout(), refresh_per_second=2, console=self.console) as live:
                self.live = live
                self.status = 'running'
                
                while self.running:
                    # Process signal queue
                    self._process_signal_queue()
                    
                    # Process update queue  
                    self._process_update_queue()
                    
                    # Update display
                    live.update(self._create_layout())
                    
                    await asyncio.sleep(0.5)
                    
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
            
            # Determine status
            age = datetime.now() - signal['timestamp']
            if age > timedelta(hours=2):
                status = "[dim]EXPIRED[/]"
            elif i == len(recent_signals) - 1:  # Latest signal
                status = "[bold yellow]● NEW[/]"
            else:
                status = "[dim]ACTIVE[/]"
                
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
        
        stats_table.add_row("Total Signals", str(len(self.signals)))
        stats_table.add_row("Active", str(len([s for s in self.signals if datetime.now() - s['timestamp'] < timedelta(hours=2)])))
        stats_table.add_row("HTF Bias", self.htf_bias.title())
        stats_table.add_row("Avg RR", f"{self.stats['avg_rr']:.1f}")
        
        # Controls
        controls_text = Text("\n🎛️ Controls:\n", style="bold cyan")
        controls_text.append("[T] Test Signal\n", style="white")
        controls_text.append("[P] Pause/Resume\n", style="white") 
        controls_text.append("[C] Clear Signals\n", style="white")
        controls_text.append("[S] Settings\n", style="white")
        controls_text.append("[Q] Quit\n", style="white")
        controls_text.append("[H] Help", style="white")
        
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
        
    def _process_signal_queue(self):
        """Process signal queue"""
        try:
            while True:
                msg_type, data = self.signal_queue.get_nowait()
                if msg_type == 'signal':
                    self.signals.append(data)
                    self._show_signal_notification(data)
                    self._update_stats()
        except queue.Empty:
            pass
            
    def _process_update_queue(self):
        """Process update queue"""
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                if msg_type == 'update':
                    # Update current state
                    engine_status = self.engine.get_status()
                    self.current_price = engine_status.get('current_price', 0.0) or 0.0
                    self.htf_bias = engine_status.get('htf_bias', 'neutral')
        except queue.Empty:
            pass
            
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
        try:
            while self.running:
                if keyboard and keyboard.is_pressed('q'):
                    self.running = False
                    break
                elif keyboard and keyboard.is_pressed('t'):
                    self._test_signal()
                elif keyboard and keyboard.is_pressed('c'):
                    self.signals.clear()
                elif keyboard and keyboard.is_pressed('p'):
                    # Toggle pause (placeholder)
                    pass
                    
                asyncio.sleep(0.1)
        except Exception as e:
            pass
            
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
        
async def run_live_monitor(symbol: str, config: Dict):
    """Run the live monitor"""
    monitor = LiveMonitorTUI(symbol, config)
    await monitor.start()
