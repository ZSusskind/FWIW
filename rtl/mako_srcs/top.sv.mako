############################################################
## Generator file for top.sv
############################################################
<%!
    import math
%>\

module device_top (
    clk, rst, inp_vld, outp_vld, stall,
    inp,
    outp
);
<%
    input_size = config["bus_width"]
    output_size = math.ceil(math.log2(info["num_classes"]))
    sample_size = info["num_inputs"]
%>\
    input  clk;
    input  rst;
    input  inp_vld;
    output outp_vld;
    output stall;

    input  [${input_size-1}:0]     inp;
    output [${output_size-1}:0]    outp;

    wire                      interface_outp_vld;
    wire [${sample_size-1}:0] interface_outp;

    wire model_stall;


    device_interface ifc (
        .clk(clk), .rst(rst),
        .inp_vld(inp_vld), .outp_vld(interface_outp_vld),
        .inp_stall(model_stall), .outp_stall(stall),
        .inp(inp),
        .outp(interface_outp)
    );

    wisard_ensemble model (
        .clk(clk), .rst(rst),
        .inp_vld(interface_outp_vld), .outp_vld(outp_vld),
        .stall(model_stall),
        .inp(interface_outp),
        .outp(outp)
    );
endmodule

