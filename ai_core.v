// =============================================================================
// Module: ai_core
// Description: Synthesizable 16-bit Matrix Multiply-Accumulate (MAC) Acceleration
//              Engine tailored for RISC-V Custom Coprocessor Interfaces & AI Workloads.
//
// Features:
//   - 16-bit signed INT16 / Fixed-Point multiply-accumulate unit
//   - High-throughput streaming data interface for weights and activations
//   - 40-bit accumulator precision to prevent overflow over deep inner-products
//   - Configurable saturation, rounding, and activation (ReLU / Linear)
//   - RISC-V Coprocessor Command Interface (Custom-0 instruction decode)
//   - Fully synchronous, synthesizable RTL architecture
// =============================================================================

`timescale 1ns / 1ps

module ai_core #(
    parameter DATA_WIDTH = 16,
    parameter MULT_WIDTH = 32,
    parameter ACC_WIDTH  = 40
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // -------------------------------------------------------------------------
    // RISC-V Coprocessor Interface (Decoded Instruction / Control)
    // -------------------------------------------------------------------------
    input  wire                   cmd_valid,
    output wire                   cmd_ready,
    input  wire [6:0]             cmd_opcode,     // 7'b0001011 (Custom-0)
    input  wire [2:0]             cmd_funct3,    // Sub-operation select
    input  wire [6:0]             cmd_funct7,    // Function modifier
    input  wire [31:0]            cmd_rs1_data,   // Scalar/Operand 1 or Weight
    input  wire [31:0]            cmd_rs2_data,   // Scalar/Operand 2 or Activation
    output reg  [31:0]            cmd_rd_data,    // Result returned to RISC-V register
    output reg                    cmd_rd_valid,

    // -------------------------------------------------------------------------
    // Direct High-Throughput Streaming Matrix / Tensor Interface
    // -------------------------------------------------------------------------
    input  wire                   stream_valid,
    output wire                   stream_ready,
    input  wire signed [DATA_WIDTH-1:0] stream_weight,     // 16-bit signed weight
    input  wire signed [DATA_WIDTH-1:0] stream_act,        // 16-bit signed activation
    input  wire                   stream_last,       // End of row / vector dot-product
    input  wire                   stream_clr_acc,    // Explicit clear accumulator

    // -------------------------------------------------------------------------
    // Engine Output Interface
    // -------------------------------------------------------------------------
    output reg                    out_valid,
    input  wire                   out_ready,
    output reg  signed [31:0]     out_data_32,       // Full 32-bit accumulator result
    output reg  signed [15:0]     out_data_sat,      // 16-bit saturated / activated
    output reg                    out_last,
    output wire                   busy,
    output reg                    overflow_flag
);

    // =========================================================================
    // RISC-V Custom Coprocessor Opcodes & Functions
    // =========================================================================
    localparam OPCODE_CUSTOM_0   = 7'b0001011;

    localparam FUNCT3_CLEAR      = 3'b000; // Reset accumulator & internal states
    localparam FUNCT3_MAC_STEP   = 3'b001; // Direct MAC via rs1, rs2
    localparam FUNCT3_STREAM_CFG = 3'b010; // Configure streaming mode / ReLU / shift
    localparam FUNCT3_READ_ACC   = 3'b011; // Read current 32-bit accumulator
    localparam FUNCT3_READ_SAT   = 3'b100; // Read 16-bit saturated/ReLU result

    // =========================================================================
    // Control & Configuration Registers
    // =========================================================================
    reg        cfg_relu_en;
    reg [4:0]  cfg_shift_right;
    reg        streaming_active;

    // Direct interface multiplexing: either from coprocessor or streaming bus
    wire signed [DATA_WIDTH-1:0] selected_weight;
    wire signed [DATA_WIDTH-1:0] selected_act;
    wire                         mac_enable;
    wire                         clr_accumulator;
    wire                         step_last;

    wire is_coproc_mac = cmd_valid && (cmd_opcode == OPCODE_CUSTOM_0) && (cmd_funct3 == FUNCT3_MAC_STEP);
    wire is_coproc_clr = cmd_valid && (cmd_opcode == OPCODE_CUSTOM_0) && (cmd_funct3 == FUNCT3_CLEAR);

    assign cmd_ready = !busy || (cmd_opcode == OPCODE_CUSTOM_0 && cmd_funct3 != FUNCT3_MAC_STEP);
    assign stream_ready = (!busy || out_ready) && !is_coproc_mac;

    assign selected_weight = is_coproc_mac ? cmd_rs1_data[DATA_WIDTH-1:0] : stream_weight;
    assign selected_act    = is_coproc_mac ? cmd_rs2_data[DATA_WIDTH-1:0] : stream_act;
    assign mac_enable      = is_coproc_mac ? 1'b1 : (stream_valid && stream_ready);
    assign clr_accumulator = is_coproc_clr || (stream_valid && stream_clr_acc);
    assign step_last       = is_coproc_mac ? cmd_funct7[0] : (stream_valid && stream_last);

    // =========================================================================
    // Pipeline Stage 1: Operand Latch & Sign Preservation
    // =========================================================================
    reg signed [DATA_WIDTH-1:0] s1_weight;
    reg signed [DATA_WIDTH-1:0] s1_act;
    reg                         s1_valid;
    reg                         s1_clr;
    reg                         s1_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_weight <= {DATA_WIDTH{1'b0}};
            s1_act    <= {DATA_WIDTH{1'b0}};
            s1_valid  <= 1'b0;
            s1_clr    <= 1'b0;
            s1_last   <= 1'b0;
        end else begin
            s1_valid <= mac_enable;
            s1_clr   <= clr_accumulator;
            s1_last  <= step_last;
            if (mac_enable) begin
                s1_weight <= selected_weight;
                s1_act    <= selected_act;
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 2: 16-bit Signed Multiplier
    // =========================================================================
    reg signed [MULT_WIDTH-1:0] s2_product;
    reg                         s2_valid;
    reg                         s2_clr;
    reg                         s2_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_product <= {MULT_WIDTH{1'b0}};
            s2_valid   <= 1'b0;
            s2_clr     <= 1'b0;
            s2_last    <= 1'b0;
        end else begin
            s2_valid <= s1_valid;
            s2_clr   <= s1_clr;
            s2_last  <= s1_last;
            if (s1_valid) begin
                s2_product <= s1_weight * s1_act;
            end else begin
                s2_product <= {MULT_WIDTH{1'b0}};
            end
        end
    end

    // =========================================================================
    // Pipeline Stage 3: 40-bit Accumulator with Sign-Extension
    // =========================================================================
    reg signed [ACC_WIDTH-1:0] accumulator;
    wire signed [ACC_WIDTH-1:0] product_ext = {{ (ACC_WIDTH-MULT_WIDTH){s2_product[MULT_WIDTH-1]} }, s2_product};

    reg s3_valid;
    reg s3_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator   <= {ACC_WIDTH{1'b0}};
            overflow_flag <= 1'b0;
            s3_valid      <= 1'b0;
            s3_last       <= 1'b0;
        end else begin
            s3_valid <= s2_valid;
            s3_last  <= s2_last;
            if (s2_clr) begin
                if (s2_valid) begin
                    accumulator <= product_ext;
                end else begin
                    accumulator <= {ACC_WIDTH{1'b0}};
                end
                overflow_flag <= 1'b0;
            end else if (s2_valid) begin
                accumulator <= accumulator + product_ext;
            end
        end
    end

    // =========================================================================
    // Quantization, Scaling, ReLU, & Saturation Logic
    // =========================================================================
    wire signed [ACC_WIDTH-1:0] shifted_acc = accumulator >>> cfg_shift_right;
    reg signed [31:0] raw_out_32;
    reg signed [15:0] saturated_out_16;

    always @(*) begin
        // Full 32-bit truncation / clamp
        if (shifted_acc > 40'sd2147483647)
            raw_out_32 = 32'sd2147483647;
        else if (shifted_acc < -40'sd2147483648)
            raw_out_32 = -32'sd2147483648;
        else
            raw_out_32 = shifted_acc[31:0];

        // 16-bit saturated activation (with optional ReLU)
        if (cfg_relu_en && shifted_acc < 0) begin
            saturated_out_16 = 16'sh0000;
        end else if (shifted_acc > 40'sd32767) begin
            saturated_out_16 = 16'sd32767;
        end else if (shifted_acc < -40'sd32768) begin
            saturated_out_16 = -16'sd32768;
        end else begin
            saturated_out_16 = shifted_acc[15:0];
        end
    end

    // =========================================================================
    // Output Registers & Handshaking
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid    <= 1'b0;
            out_last     <= 1'b0;
            out_data_32  <= 32'sd0;
            out_data_sat <= 16'sd0;
        end else begin
            if (s3_valid && s3_last) begin
                out_valid    <= 1'b1;
                out_last     <= 1'b1;
                out_data_32  <= raw_out_32;
                out_data_sat <= saturated_out_16;
            end else if (out_ready && out_valid) begin
                out_valid <= 1'b0;
                out_last  <= 1'b0;
            end
        end
    end

    assign busy = s1_valid || s2_valid || s3_valid;

    // =========================================================================
    // RISC-V Coprocessor Instruction Handling
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_rd_data      <= 32'h0;
            cmd_rd_valid     <= 1'b0;
            cfg_relu_en      <= 1'b0;
            cfg_shift_right  <= 5'd0;
            streaming_active <= 1'b0;
        end else begin
            cmd_rd_valid <= 1'b0;
            if (cmd_valid && (cmd_opcode == OPCODE_CUSTOM_0)) begin
                case (cmd_funct3)
                    FUNCT3_CLEAR: begin
                        cmd_rd_data  <= 32'h0;
                        cmd_rd_valid <= 1'b1;
                    end
                    FUNCT3_STREAM_CFG: begin
                        cfg_relu_en      <= cmd_rs1_data[0];
                        cfg_shift_right  <= cmd_rs1_data[5:1];
                        streaming_active <= cmd_rs2_data[0];
                        cmd_rd_data      <= 32'h00000001; // Success
                        cmd_rd_valid     <= 1'b1;
                    end
                    FUNCT3_READ_ACC: begin
                        cmd_rd_data  <= raw_out_32;
                        cmd_rd_valid <= 1'b1;
                    end
                    FUNCT3_READ_SAT: begin
                        cmd_rd_data  <= {{16{saturated_out_16[15]}}, saturated_out_16};
                        cmd_rd_valid <= 1'b1;
                    end
                    FUNCT3_MAC_STEP: begin
                        // Acknowledged via busy / pipeline
                        cmd_rd_valid <= 1'b0;
                    end
                    default: begin
                        cmd_rd_data  <= 32'h0;
                        cmd_rd_valid <= 1'b0;
                    end
                endcase
            end
        end
    end

endmodule
