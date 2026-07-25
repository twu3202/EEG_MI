# PORTABLE COPY — synced from src/ (edit src/, then re-run sync). Flat-folder imports.
#!/usr/bin/env python
"""Network self-check — can I reach the CAP and the INTERNET at the same time?

Joining the cap's Wi-Fi AP (ESPBCI) normally steals the default route, so the machine
loses internet while recording. The fix is to give the *internet* to a second interface
(iPhone USB tethering / Ethernet dongle) and let Wi-Fi carry ONLY the 192.168.4.0/24
subnet. This script verifies that split actually holds:

  * lists every active interface + IP, and which one owns the default route
  * checks the route to the cap (192.168.4.1) goes out the Wi-Fi interface
  * probes the cap (ICMP + the real UDP command port)
  * probes the internet (TCP 443 to a public DNS resolver — no data sent)

  python src/acquisition/net_check.py
  python src/acquisition/net_check.py --watch        # re-check every 3 s

Read-only: it never changes your network settings. If something is wrong it prints the
exact fix (see docs/network_setup.md).
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import time

CAP_IP = "192.168.4.1"
CAP_PORT = 8086          # board command port
LOCAL_PORT = 2244        # the board streams data back to us here
WANT_LOCAL_IP = "192.168.4.2"
INTERNET_PROBES = [("1.1.1.1", 443), ("8.8.8.8", 443)]

G, Y, R, B, X = "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=6).stdout
    except Exception:
        return ""


def interfaces():
    """[(service, device, ip)] for every configured service that currently has an IP."""
    out, cur = [], None
    for line in sh("networksetup -listnetworkserviceorder").splitlines():
        line = line.strip()
        if line.startswith("(") and ")" in line and "Hardware Port" not in line:
            cur = line.split(")", 1)[1].strip()
        elif line.startswith("(Hardware Port") and cur:
            dev = line.rstrip(")").split("Device:")[-1].strip()
            ip = sh(f"ipconfig getifaddr {dev}").strip()
            if ip:
                out.append((cur, dev, ip))
            cur = None
    return out


def default_route():
    txt = sh("route -n get default")
    gw = dev = None
    for ln in txt.splitlines():
        if "gateway:" in ln:
            gw = ln.split(":", 1)[1].strip()
        if "interface:" in ln:
            dev = ln.split(":", 1)[1].strip()
    return dev, gw


def route_to(ip):
    txt = sh(f"route -n get {ip}")
    for ln in txt.splitlines():
        if "interface:" in ln:
            return ln.split(":", 1)[1].strip()
    return None


def tcp_ok(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ping_ok(host, n=1, timeout=2):
    return subprocess.run(f"ping -c {n} -W {timeout*1000:.0f} {host}",
                          shell=True, capture_output=True).returncode == 0


def wifi_takes_dhcp_router():
    """Is the Wi-Fi service on DHCP? Then joining ESPBCI installs 192.168.4.1 as a
    competing DEFAULT GATEWAY — which is what makes Clash/mihomo bind its outbound
    connections to the cap's dead-end Wi-Fi and stop working."""
    txt = sh('networksetup -getinfo "Wi-Fi"')
    return "DHCP Configuration" in txt, txt


def system_proxy():
    """(host, port, services_with_proxy) from macOS per-service proxy settings."""
    host = port = None
    svcs = []
    for svc, _dev, _ip in interfaces() or []:
        t = sh(f'networksetup -getwebproxy "{svc}"')
        if "Enabled: Yes" in t:
            svcs.append(svc)
            for ln in t.splitlines():
                if ln.startswith("Server:"):
                    host = ln.split(":", 1)[1].strip()
                if ln.startswith("Port:"):
                    port = int(ln.split(":", 1)[1].strip() or 0)
    return host, port, svcs


def proxy_tunnel_ok(phost, pport, target="www.cloudflare.com", tport=443, timeout=4.0):
    """Ask the proxy to CONNECT somewhere — proves the proxy's OUTBOUND path works."""
    try:
        s = socket.create_connection((phost, pport), timeout=timeout)
    except OSError:
        return None                                    # proxy not even listening
    try:
        s.sendall(f"CONNECT {target}:{tport} HTTP/1.1\r\n"
                  f"Host: {target}:{tport}\r\n\r\n".encode())
        return b" 200 " in s.recv(256)
    except OSError:
        return False
    finally:
        s.close()


def udp_port_free(port):
    """Is our data port already taken (e.g. the vendor main_ui is running)?"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def check(verbose=True):
    ifs = interfaces()
    dev, gw = default_route()
    cap_dev = route_to(CAP_IP)
    have_cap_route = cap_dev is not None and gw != CAP_IP or cap_dev is not None
    local_ips = [ip for _, _, ip in ifs]
    on_cap_subnet = any(ip.startswith("192.168.4.") for ip in local_ips)
    cap_ip_ok = WANT_LOCAL_IP in local_ips

    cap_ping = ping_ok(CAP_IP) if on_cap_subnet else False
    # internet: TCP first; fall back to ICMP + "default gateway isn't the cap" so the check
    # still works where outbound TCP is filtered (corporate proxy, sandbox, firewall).
    net_ok = any(tcp_ok(h, p) for h, p in INTERNET_PROBES)
    if not net_ok:
        net_ok = any(ping_ok(h) for h, _ in INTERNET_PROBES) or (
            bool(gw) and gw != CAP_IP and not gw.startswith("192.168.4.") and ping_ok(gw))
    port_free = udp_port_free(LOCAL_PORT)

    if verbose:
        print(f"\n{B}接口 interfaces{X}")
        for svc, d, ip in ifs:
            tag = f"  {G}← 默认路由(上网){X}" if d == dev else ""
            tag += f"  {G}← 帽子{X}" if ip.startswith("192.168.4.") else ""
            print(f"  {svc:20s} {d:8s} {ip:15s}{tag}")
        print(f"  默认路由: {dev or '—'} → {gw or '—'}")

        print(f"\n{B}检查 checks{X}")
        def row(ok, label, hint="", warn=False):
            c = G if ok else (Y if warn else R)
            print(f"  {c}{'✔' if ok else ('!' if warn else '✘')}{X} {label}" + (f"   {hint}" if not ok else ""))

        row(on_cap_subnet, f"在帽子子网 (192.168.4.x)", "→ 连上 ESPBCI Wi-Fi")
        row(cap_ip_ok, f"本机 IP = {WANT_LOCAL_IP}",
            f"→ 当前 {[i for i in local_ips if i.startswith('192.168.4.')] or '无'};"
            " 板子只往 .2 发数据,重连或设静态 IP", warn=on_cap_subnet)
        row(cap_ping, f"能 ping 通帽子 {CAP_IP}", "→ 检查 ESPBCI 连接")
        row(net_ok, "能上网 (TCP 443)",
            "→ 插 iPhone USB / 网线,并把它排到 Wi-Fi 之上(见下方)")
        row(port_free, f"UDP {LOCAL_PORT} 空闲", "→ 关掉厂商 main_ui.exe 或其它占用的程序", warn=True)
        if on_cap_subnet and cap_dev and dev and cap_dev == dev and not net_ok:
            print(f"  {Y}!{X} 默认路由被帽子抢走了 —— 这就是断网的原因")

        # ---- proxy (Clash/mihomo) ----
        phost, pport, psvcs = system_proxy()
        if phost:
            tunnel = proxy_tunnel_ok(phost, pport)
            print(f"\n{B}代理 proxy{X}  ({phost}:{pport}, 已设于 {len(psvcs)} 个服务)")
            row(tunnel is not None, "代理端口在监听", "→ Clash 没运行?")
            row(bool(tunnel), "通过代理能出网 (CONNECT)",
                "→ 代理的出站走错网卡了(多半被 ESPBCI 抢走)")
            dhcp, _ = wifi_takes_dhcp_router()
            row(not dhcp, "Wi-Fi 不会抢默认网关",
                "→ Wi-Fi 是 DHCP:连 ESPBCI 会拿到网关 192.168.4.1,"
                "Clash 出站可能被绑到这个死路上", warn=True)
            if on_cap_subnet and tunnel is False:
                print(f"  {Y}!{X} 这就是「能上网但代理失效」的典型表现")

        both = cap_ping and net_ok
        print(f"\n{B}结论{X}: " + (f"{G}✅ 帽子 + 互联网 同时在线,可以一边采集一边联网{X}" if both
              else f"{Y}⚠ 只有帽子(无网){X}" if cap_ping
              else f"{Y}⚠ 只有网(帽子未连){X}" if net_ok else f"{R}✘ 两个都不通{X}"))
        if cap_ping and not net_ok:
            print(f"""
{B}修复(二选一){X}
  A. iPhone USB 共享:数据线连 iPhone → 设置·个人热点 打开 →
     系统设置 · 网络 · ⋯ · 服务顺序,把 {B}iPhone USB 拖到 Wi-Fi 之上{X}
  B. 有线网卡(USB-C 转以太网)同理,把 {B}Ethernet 拖到 Wi-Fi 之上{X}
  之后:Wi-Fi 只负责 192.168.4.x(帽子),上网走另一个口。详见 docs/network_setup.md""")
    return dict(interfaces=ifs, default_dev=dev, cap=cap_ping, internet=net_ok,
                cap_ip_ok=cap_ip_ok, port_free=port_free)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true", help="re-check every 3 s")
    args = ap.parse_args()
    if not args.watch:
        check(); return
    try:
        while True:
            print("\033[2J\033[H", end="")
            check()
            time.sleep(3)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
