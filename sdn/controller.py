import ipaddress
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_0
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import ipv4, packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib import hub
from ryu.lib.mac import haddr_to_str

import numpy as np
import pickle
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler

SERVER_IP = "10.0.0.2"  # host h1
POLL_INTERVAL_S = 3  # Poll every 10 seconds
TARGET_SWITCH_DPIDS = {1, 2} # DPIDs of s1 and s2 (Mininet usually assigns 1, 2)

scaler_params = {
    "min_": [0.0, -9.985321577281397e-06, 0.0, 0.0, 0.0, 0.0],
    "scale_": [8.333333333333334e-09, 9.985321577281397e-06, 0.000217296827466319, 3.3967391304347827e-10, 2.5e-07, 0.00024851232105075894],
    "data_min_": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "data_max_": [120000000.0, 100148.0, 4602.0, 2944000000.0, 4000000.0, 4023.9453551912575],
    "data_range_": [120000000.0, 100147.0, 4602.0, 2944000000.0, 4000000.0, 4023.9453551912575],
    "feature_range": (0, 1)
}

scaler = MinMaxScaler()
scaler.min_ = np.array(scaler_params["min_"])
scaler.scale_ = np.array(scaler_params["scale_"])
scaler.data_min_ = np.array(scaler_params["data_min_"])
scaler.data_max_ = np.array(scaler_params["data_max_"])
scaler.data_range_ = np.array(scaler_params["data_range_"])
scaler.feature_range = scaler_params["feature_range"]
scaler.n_features_in_ = len(scaler_params["min_"])


# Transformer Encoder for DDoS Detection
class DDoSTransformer(nn.Module):
    def __init__(
        self, input_dim, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1
    ):
        super(DDoSTransformer, self).__init__()

        # Embedding layer to project input features
        self.input_projection = nn.Linear(input_dim, dim_feedforward)

        # Positional encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward), nn.ReLU(), nn.Dropout(dropout)
        )

        # Transformer encoder layers (simplified from full transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Linear(dim_feedforward, 2)  # Binary classification

    def forward(self, x):
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Project input
        x = self.input_projection(x)
        
        # Reshape for transformer if needed (add sequence dimension)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, features]
            
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling across sequence dimension
        x = x.mean(dim=1) if len(x.shape) > 2 else x
        
        # Classification output
        x = self.output_layer(x)

        return x


class ModelInference:
    def __init__(self, model_path, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = DDoSTransformer(input_dim=6)

        # Load the model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_data):
        # Ensure input is a tensor and on the correct device
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        input_data = input_data.to(self.device)

        # Add batch dimension if needed
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_data)

        return output


class Switch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Switch, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {} # To store datapath objects {dpid: datapath}
        self.blacklist = []
        self.monitor_thread = hub.spawn(self._monitor) # Start the polling thread
        self.logger.info("Monitoring initialized. Polling interval: %d s", POLL_INTERVAL_S)

    # Modified add_flow
    def add_flow(self, datapath, in_port, actions,
                 dl_type=None, # Optional: to specify packet type (e.g., IP, ARP)
                 nw_src=None, nw_dst=None, # For IP addresses
                 dl_src=None, dl_dst=None): # For MAC addresses (can be used as fallback or for non-IP)
        if nw_src in self.blacklist:
            self.logger.warning("Blacklisted IP %s attempted to add flow. Ignoring.", nw_src)
            return

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match_args = {'in_port': in_port}

        if dl_type is not None:
            match_args['dl_type'] = dl_type

        if nw_src is not None and nw_dst is not None and dl_type == ether_types.ETH_TYPE_IP:
            # For IP matching, nw_src and nw_dst are typically integers in OF1.0
            try:
                match_args['nw_src'] = int(ipaddress.ip_address(nw_src))
                match_args['nw_dst'] = int(ipaddress.ip_address(nw_dst))
                self.logger.debug(f"Adding L3 flow: in_port={in_port}, "
                                 f"nw_src={nw_src}({match_args['nw_src']}), "
                                 f"nw_dst={nw_dst}({match_args['nw_dst']})")
            except ValueError:
                self.logger.error(f"Invalid IP address format for flow: src={nw_src}, dst={nw_dst}")
                return # Don't add flow if IPs are bad
        elif dl_src is not None and dl_dst is not None:
            # Fallback or specific L2 flow
            match_args['dl_src'] = haddr_to_bin(dl_src)
            match_args['dl_dst'] = haddr_to_bin(dl_dst)
            self.logger.debug(f"Adding L2 flow: in_port={in_port}, "
                                 f"dl_src={dl_src}, dl_dst={dl_dst}")
        else:
            self.logger.warning("add_flow called with insufficient L2/L3 match criteria.")
            return


        match = parser.OFPMatch(**match_args)

        mod = parser.OFPFlowMod(
            datapath=datapath,
            match=match,
            cookie=0, # You can use cookie to identify flow types
            command=ofproto.OFPFC_ADD,
            idle_timeout=60,  # Good practice to add timeouts
            hard_timeout=60, # Good practice to add timeouts
            priority=ofproto.OFP_DEFAULT_PRIORITY + 1, # Higher priority for specific flows
            flags=ofproto.OFPFF_SEND_FLOW_REM,
            actions=actions,
        )
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.in_port # Corrected: get in_port from msg.match or msg directly for OF1.0

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if not eth: # Should not happen with Ethernet networks
            return

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return # Ignore LLDP

        dst_mac = eth.dst
        src_mac = eth.src
        dpid = datapath.id

        self.mac_to_port.setdefault(dpid, {})

        # Standard MAC learning
        if self.mac_to_port[dpid].get(src_mac) != in_port:
            self.logger.info("Learning MAC %s on dpid %s, port %s", src_mac, dpid, in_port)
            self.mac_to_port[dpid][src_mac] = in_port

        # Determine output port based on L2 MAC destination
        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            # If destination MAC is unknown (e.g., first packet to dest, or ARP broadcast)
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # --- Install flow based on L3 if IP, otherwise L2 for ARP ---
        if out_port != ofproto.OFPP_FLOOD: # Only install specific flows if dst MAC is known
            ipv4_pkt = pkt.get_protocol(ipv4.ipv4)

            if ipv4_pkt:
                src_ip = ipv4_pkt.src
                dst_ip = ipv4_pkt.dst
                # self.logger.info("Packet in dpid %s: ETH %s->%s, IP %s->%s, in_port %s",
                                   #                dpid, src_mac, dst_mac, src_ip, dst_ip, in_port)
                # Install L3 flow
                self.add_flow(datapath, in_port, actions,
                              dl_type=ether_types.ETH_TYPE_IP,
                              nw_src=src_ip, nw_dst=dst_ip)
            else:
                # Other L2 traffic, install general L2 flow if desired, or rely on PacketOut
                # self.logger.info("Other L2 Packet in dpid %s: %s->%s, in_port %s. Will be processed by PacketOut.",
                                   # dpid, src_mac, dst_mac, in_port)
                # If you wanted generic L2 flows for known MAC pairs:
                # self.add_flow(datapath, in_port, actions,
                #                dl_src=src_mac, dl_dst=dst_mac)
                pass

        data = None
        # Send the current packet out
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)

    def delete_flow(self, datapath, nw_src):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch(nw_src=int(ipaddress.ip_address(nw_src)), dl_type=ether_types.ETH_TYPE_IP)
        mod = parser.OFPFlowMod(
            datapath=datapath,
            match=match,
            cookie=0,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_NONE,
            priority=ofproto.OFP_DEFAULT_PRIORITY + 1,
        )
        datapath.send_msg(mod)
        self.logger.info("Deleted flow for blacklisted IP: %s", nw_src)

    def add_drop_flow(self, datapath, nw_src):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(dl_type=ether_types.ETH_TYPE_IP,
                                nw_src=int(ipaddress.ip_address(nw_src)))
        mod = parser.OFPFlowMod(
            datapath=datapath,
            match=match,
            cookie=0,
            command=ofproto.OFPFC_ADD,
            idle_timeout=60,
            hard_timeout=120,
            priority=ofproto.OFP_DEFAULT_PRIORITY + 10,
            actions=[]  # No actions => drop
        )
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Handle switch features event.
        Store datapath object and install table-miss flow entry (good practice).
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser # Corrected: get parser

        # Store datapath for later use by the monitor
        self.datapaths[datapath.id] = datapath
        self.logger.info("Switch connected: dpid=%s", datapath.id)

        # Install table-miss flow entry (sends unmatched packets to controller)
        # This is generally good practice, though your original relies on default.
        # For OF1.0, this is how you'd typically do a "send to controller" for misses.
        # However, OF1.0 doesn't have a direct "match-all" as cleanly as OF1.3.
        # Often, OF1.0 switches default to sending to controller if no flow matches.
        # If you want to be explicit, you'd install a low-priority wildcard flow.
        # For simplicity and to stick close to your original, I'll omit explicit
        # table-miss installation here, assuming the switch's default behavior.
        # If issues arise, add:
        # match = parser.OFPMatch() # This might need to be more specific in OF1.0
                                  # or rely on a very low priority flow.
        # actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER)]
        # mod = parser.OFPFlowMod(datapath=datapath, match=match, cookie=0,
        #                         command=ofproto.OFPFC_ADD, idle_timeout=0,
        #                         hard_timeout=0, priority=0, # Lowest priority
        #                         actions=actions)
        # datapath.send_msg(mod)


    def _monitor(self):
        """
        Periodically request flow statistics from target switches.
        """
        while True:
            for dpid in TARGET_SWITCH_DPIDS:
                if dpid in self.datapaths:
                    datapath = self.datapaths[dpid]
                    self.logger.info("Requesting flow stats from dpid=%s", dpid)
                    self._request_flow_stats(datapath)
            hub.sleep(POLL_INTERVAL_S)

    def _request_flow_stats(self, datapath):
        """
        Send a flow stats request to the specified datapath.
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Request stats for all flows in all tables
        # For OF1.0, match_fields for "all" can be tricky.
        # An empty match OFPMatch() often works, or specific wildcards.
        # For flow stats, we also specify table_id and out_port.
        match = parser.OFPMatch() # Match all flows
        req = parser.OFPFlowStatsRequest(datapath,
                                         flags=0, # Standard flags
                                         match=match,
                                         table_id=0xff, # All tables
                                         out_port=ofproto.OFPP_NONE) # OFPP_NONE for all ports in OF1.0
                                                                    # (or OFPP_ANY for OF1.3+)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        ofproto = ev.msg.datapath.ofproto # Get ofproto for constants

        active_flows = [flow for flow in body if flow.priority > 0] # Filter example
        sorted_flows = sorted(active_flows,
                              key=lambda flow: (flow.priority,),
                              reverse=True)

        backward_packets = {}   # Server to host packets
        forward_packets = {}
        for stat in sorted_flows:
            match_fields = stat.match
            wildcards = match_fields['wildcards'] # For OF1.0, to check if a field was part of match
            nw_src_int = match_fields['nw_src']
            src_ip_str = str(ipaddress.ip_address(nw_src_int)) if not (wildcards & ofproto.OFPFW_NW_SRC_MASK) and nw_src_int is not None else "ANY"
            nw_dst_int = match_fields['nw_dst']
            dst_ip_str = str(ipaddress.ip_address(nw_dst_int)) if not (wildcards & ofproto.OFPFW_NW_DST_MASK) and nw_dst_int is not None else "ANY"
            if src_ip_str == SERVER_IP:
                backward_packets[dst_ip_str] = {
                    "duration": stat.duration_sec,
                    "bytes": stat.byte_count,
                    "n_packets": stat.packet_count,
                }
            else:
                forward_packets[src_ip_str] = {
                    "duration": stat.duration_sec,
                    "bytes": stat.byte_count,
                    "n_packets": stat.packet_count,
                }

        for src_ip in forward_packets:
            if (
                forward_packets[src_ip]["duration"] == 0 or
                forward_packets[src_ip]["n_packets"] == 0
            ):
                continue

            duration = forward_packets[src_ip]["duration"] * 1e6
            self.logger.info("duration: %.8f", duration)
            total_fwd_packets = forward_packets[src_ip]["n_packets"]
            self.logger.info("fwd packets: %d", total_fwd_packets)
            total_bwd_packets = 0 

            flow_bytes_p_sec = forward_packets[src_ip]["bytes"] / duration
            self.logger.info("flow bytes p sec: %.8f", flow_bytes_p_sec)

            flow_packets_p_sec = total_fwd_packets / duration
            self.logger.info("flow packets p sec: %.8f", flow_packets_p_sec)
            total_packet_len_mean = forward_packets[src_ip]["bytes"] / total_fwd_packets
            self.logger.info("total packet length mean: %.8f", total_packet_len_mean)

            # self.logger.info("%s", [duration, total_fwd_packets, total_bwd_packets, flow_bytes_p_sec, flow_packets_p_sec, total_packet_len_mean])
            input_data = np.array([duration, total_fwd_packets, total_bwd_packets, flow_bytes_p_sec, flow_packets_p_sec, total_packet_len_mean])
            self.logger.info("%s", input_data)
            # input_data = np.array([3, 2, 0, 0, 666666, 0])
            # input_data = np.array([30000000, 2, 0, 65, 0.06667, 976.0])

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.device = torch.device('cpu')
            model_path = "./best_model.pth"
            inference = ModelInference(model_path, self.device)

            input_normalized = scaler.transform(input_data.reshape(1, -1))
            input_tensor = torch.FloatTensor(input_normalized).to("cpu")
            result = inference.predict(input_tensor)
            prediction = result.argmax(dim=1).item()
            if prediction == 1:
                self.logger.info("\n\nDDoS attack detected from IP: %s", src_ip)
                self.add_drop_flow(ev.msg.datapath, src_ip)
                if src_ip not in self.blacklist:
                    self.logger.info("Adding packet drop rule to switch for IP: %s", src_ip)
                    self.blacklist.append(src_ip)

        return

        self.logger.info("Received flow stats from dpid=%s:", dpid)
        self.logger.info( # Updated header
            "---------------- -------- -------- -------- -------- ------------------- ------------------- ------------------- ------------------- -----------"
        )
        self.logger.info(
            "table_id priority dl_type  nw_src          nw_dst              actions     pkt_count       byte_count          duration"
        )
        self.logger.info(
            "---------------- -------- -------- -------- -------- ------------------- ------------------- ------------------- ------------------- -----------"
        )


        if not sorted_flows:
            self.logger.info(" (No active flows with priority > 0 found for dpid %s)", dpid)

        for stat in sorted_flows:
            # --- Get Match Fields Safely ---
            match_fields = stat.match
            wildcards = match_fields['wildcards'] # For OF1.0, to check if a field was part of match
            nw_src_int = match_fields['nw_src']
            src_ip_str = str(ipaddress.ip_address(nw_src_int)) if not (wildcards & ofproto.OFPFW_NW_SRC_MASK) and nw_src_int is not None else "ANY"
            nw_dst_int = match_fields['nw_dst']
            dst_ip_str = str(ipaddress.ip_address(nw_dst_int)) if not (wildcards & ofproto.OFPFW_NW_DST_MASK) and nw_dst_int is not None else "ANY"

            dl_type_val = match_fields['dl_type']
            if isinstance(dl_type_val, int): # dl_type is usually an int
                 if dl_type_val == ether_types.ETH_TYPE_IP: dl_type_str = "IP"
                 elif dl_type_val == ether_types.ETH_TYPE_ARP: dl_type_str = "ARP"
                 else: dl_type_str = hex(dl_type_val)
            else:
                 dl_type_str = str(dl_type_val)

            # OFPFW_NW_SRC_MASK is (0x3f << OFPFW_NW_SRC_SHIFT), OFPFW_NW_SRC_SHIFT is 8. So mask is 0x3F00
            # A simpler check for nw_src often is just to see if nw_src_int exists and dl_type is IP

            actions_str = []
            for action in stat.actions:
                if action.type == ofproto_v1_0.OFPAT_OUTPUT:
                    actions_str.append(f"OUTPUT:{action.port}")
                else:
                    actions_str.append(f"TYPE:{action.type}")
            actions_display = ",".join(actions_str) if actions_str else "NONE"

            self.logger.info(
                "%8s %8d %8s %-19s %-19s %-11s %9d %10d %8s",
                stat.table_id,
                stat.priority,
                dl_type_str,
                src_ip_str,
                dst_ip_str,
                actions_display,
                stat.packet_count,
                stat.byte_count,
                stat.duration_sec,
            )

        self.logger.info(
            "---------------- -------- -------- -------- -------- ------------------- ------------------- ------------------- ------------------- -----------"
        )

