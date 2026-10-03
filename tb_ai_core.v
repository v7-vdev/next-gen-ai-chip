// =============================================================================
// Testbench: tb_ai_core
// Description: Verification testbench for the 16-bit RISC-V AI Acceleration Core.
//              Simulates streaming matrix multiplication (GEMM) operations,
//              verifies computed outputs against a golden software model,
//              and logs 'STATUS: SUCCESS' upon full verification.
// =============================================================================

`timescale 1ns / 1ps

module tb_ai_core;

    // Parameters
    localparam DATA_WIDTH = 16;
    localparam MULT_WIDTH = 32;
    localparam ACC_WIDTH  = 40;
    localparam CLK_PERIOD = 10; // 100 MHz clock

    // Clock and Reset
    reg clk;
    reg rst_n;

    // RISC-V Coprocessor Command Interface
    reg         cmd_valid;
    wire        cmd_ready;
    reg  [6:0]  cmd_opcode;
    reg  [2:0]  cmd_funct3;
    reg  [6:0]  cmd_funct7;
    reg  [31:0] cmd_rs1_data;
    reg  [31:0] cmd_rs2_data;
    wire [31:0] cmd_rd_data;
    wire        cmd_rd_valid;

    // Streaming Matrix Interface
    reg                   stream_valid;
    wire                  stream_ready;
    reg  signed [15:0]    stream_weight;
    reg  signed [15:0]    stream_act;
    reg                   stream_last;
    reg                   stream_clr_acc;

    // Output Interface
    wire                  out_valid;
    reg                   out_ready;
    wire signed [31:0]    out_data_32;
    wire signed [15:0]    out_data_sat;
    wire                  out_last;
    wire                  busy;
    wire                  overflow_flag;

    // Test tracking variables
    integer total_tests;
    integer passed_tests;
    integer failed_tests;
    integer i, j, k;

    // -------------------------------------------------------------------------
    // Instantiate Device Under Test (DUT)
    // -------------------------------------------------------------------------
    ai_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .MULT_WIDTH(MULT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_valid      (cmd_valid),
        .cmd_ready      (cmd_ready),
        .cmd_opcode     (cmd_opcode),
        .cmd_funct3     (cmd_funct3),
        .cmd_funct7     (cmd_funct7),
        .cmd_rs1_data   (cmd_rs1_data),
        .cmd_rs2_data   (cmd_rs2_data),
        .cmd_rd_data    (cmd_rd_data),
        .cmd_rd_valid   (cmd_rd_valid),
        .stream_valid   (stream_valid),
        .stream_ready   (stream_ready),
        .stream_weight  (stream_weight),
        .stream_act     (stream_act),
        .stream_last    (stream_last),
        .stream_clr_acc (stream_clr_acc),
        .out_valid      (out_valid),
        .out_ready      (out_ready),
        .out_data_32    (out_data_32),
        .out_data_sat   (out_data_sat),
        .out_last       (out_last),
        .busy           (busy),
        .overflow_flag  (overflow_flag)
    );

    // -------------------------------------------------------------------------
    // Clock Generation
    // -------------------------------------------------------------------------
    always #(CLK_PERIOD / 2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Matrix Definitions for Verification
    // Matrix A (3x4) * Matrix B (4x3) = Matrix C (3x3)
    // -------------------------------------------------------------------------
    localparam M = 3;
    localparam K_DIM = 4;
    localparam N = 3;

    reg signed [15:0] matrix_A [0:M-1][0:K_DIM-1];
    reg signed [15:0] matrix_B [0:K_DIM-1][0:N-1];
    reg signed [31:0] golden_C [0:M-1][0:N-1];

    // -------------------------------------------------------------------------
    // Task: Reset Sequence
    // -------------------------------------------------------------------------
    task reset_dut;
        begin
            rst_n          = 1'b0;
            cmd_valid      = 1'b0;
            cmd_opcode     = 7'd0;
            cmd_funct3     = 3'd0;
            cmd_funct7     = 7'd0;
            cmd_rs1_data   = 32'd0;
            cmd_rs2_data   = 32'd0;
            stream_valid   = 1'b0;
            stream_weight  = 16'd0;
            stream_act     = 16'd0;
            stream_last    = 1'b0;
            stream_clr_acc = 1'b0;
            out_ready      = 1'b1;

            #(CLK_PERIOD * 4);
            @(posedge clk);
            rst_n = 1'b1;
            #(CLK_PERIOD * 2);
        end
    endtask

    // -------------------------------------------------------------------------
    // Task: Stream Single Dot Product Vector
    // -------------------------------------------------------------------------
    task stream_dot_product;
        input integer row_idx;
        input integer col_idx;
        input integer vec_len;
        output signed [31:0] hw_result;
        integer idx;
        begin
            for (idx = 0; idx < vec_len; idx = idx + 1) begin
                @(posedge clk);
                stream_valid   <= 1'b1;
                stream_weight  <= matrix_A[row_idx][idx];
                stream_act     <= matrix_B[idx][col_idx];
                stream_clr_acc <= (idx == 0) ? 1'b1 : 1'b0;
                stream_last    <= (idx == vec_len - 1) ? 1'b1 : 1'b0;
            end

            @(posedge clk);
            stream_valid   <= 1'b0;
            stream_clr_acc <= 1'b0;
            stream_last    <= 1'b0;

            // Wait for output valid
            while (!out_valid) begin
                @(posedge clk);
            end

            hw_result = out_data_32;
            @(posedge clk);
        end
    endtask

    // -------------------------------------------------------------------------
    // Main Test Sequence
    // -------------------------------------------------------------------------
    reg signed [31:0] calculated_out;

    initial begin
        $display("------------------------------------------------------------");
        $display("Starting RISC-V AI Acceleration Core Hardware Verification ");
        $display("------------------------------------------------------------");

        // Optional VCD dump for waveform visualizer
        $dumpfile("ai_core_sim.vcd");
        $dumpvars(0, tb_ai_core);

        clk          = 1'b0;
        total_tests  = 0;
        passed_tests = 0;
        failed_tests = 0;

        reset_dut();

        // =====================================================================
        // Test Suite 1: RISC-V Coprocessor Custom Instructions
        // =====================================================================
        $display("\n[TEST 1] Testing RISC-V Custom Coprocessor Instruction Interface...");
        
        // 1.1 Clear Accumulator via Coprocessor
        @(posedge clk);
        cmd_valid    <= 1'b1;
        cmd_opcode   <= 7'b0001011; // CUSTOM_0
        cmd_funct3   <= 3'b000;     // FUNCT3_CLEAR
        cmd_rs1_data <= 32'd0;
        cmd_rs2_data <= 32'd0;
        @(posedge clk);
        cmd_valid    <= 1'b0;
        #(CLK_PERIOD * 2);

        // 1.2 Step 1: Weight = 12, Act = 15 -> Product = 180
        @(posedge clk);
        cmd_valid    <= 1'b1;
        cmd_opcode   <= 7'b0001011;
        cmd_funct3   <= 3'b001;     // FUNCT3_MAC_STEP
        cmd_funct7   <= 7'b0000000;
        cmd_rs1_data <= 32'd12;
        cmd_rs2_data <= 32'd15;
        @(posedge clk);

        // 1.3 Step 2: Weight = -5, Act = 20 -> Product = -100 (Acc = 80)
        cmd_rs1_data <= -32'd5;
        cmd_rs2_data <= 32'd20;
        @(posedge clk);
        cmd_valid    <= 1'b0;

        // Allow pipeline to settle
        #(CLK_PERIOD * 4);

        // 1.4 Read Accumulator via Coprocessor
        @(posedge clk);
        cmd_valid    <= 1'b1;
        cmd_funct3   <= 3'b011;     // FUNCT3_READ_ACC
        @(posedge clk);
        cmd_valid    <= 1'b0;
        @(posedge clk);

        total_tests = total_tests + 1;
        if (cmd_rd_data == 32'd80) begin
            $display("  -> Coprocessor MAC step & read check: PASSED (Expected: 80, Got: %0d)", $signed(cmd_rd_data));
            passed_tests = passed_tests + 1;
        end else begin
            $display("  -> Coprocessor MAC step check: FAILED (Expected: 80, Got: %0d)", $signed(cmd_rd_data));
            failed_tests = failed_tests + 1;
        end

        // =====================================================================
        // Test Suite 2: Streaming Matrix Multiplication (3x4 * 4x3)
        // =====================================================================
        $display("\n[TEST 2] Initializing Matrix Multiplication Engine Verification...");

        // Initialize Matrix A (3x4) with mixed positive and negative 16-bit values
        matrix_A[0][0] = 16'sd12;   matrix_A[0][1] = -16'sd3;  matrix_A[0][2] = 16'sd7;   matrix_A[0][3] = 16'sd1;
        matrix_A[1][0] = -16'sd4;   matrix_A[1][1] = 16'sd15;  matrix_A[1][2] = -16'sd2;  matrix_A[1][3] = 16'sd8;
        matrix_A[2][0] = 16'sd9;    matrix_A[2][1] = 16'sd0;   matrix_A[2][2] = 16'sd14;  matrix_A[2][3] = -16'sd6;

        // Initialize Matrix B (4x3)
        matrix_B[0][0] = 16'sd5;    matrix_B[0][1] = 16'sd2;   matrix_B[0][2] = -16'sd1;
        matrix_B[1][0] = -16'sd8;   matrix_B[1][1] = 16'sd4;   matrix_B[1][2] = 16'sd3;
        matrix_B[2][0] = 16'sd6;    matrix_B[2][1] = -16'sd7;  matrix_B[2][2] = 16'sd11;
        matrix_B[3][0] = 16'sd10;   matrix_B[3][1] = 16'sd1;   matrix_B[3][2] = -16'sd4;

        // Calculate Golden Reference Matrix C in software
        for (i = 0; i < M; i = i + 1) begin
            for (j = 0; j < N; j = j + 1) begin
                golden_C[i][j] = 0;
                for (k = 0; k < K_DIM; k = k + 1) begin
                    golden_C[i][j] = golden_C[i][j] + (matrix_A[i][k] * matrix_B[k][j]);
                end
            end
        end

        // Stream all matrix dot-products into the AI hardware MAC core
        for (i = 0; i < M; i = i + 1) begin
            for (j = 0; j < N; j = j + 1) begin
                total_tests = total_tests + 1;
                stream_dot_product(i, j, K_DIM, calculated_out);

                if (calculated_out === golden_C[i][j]) begin
                    $display("  [PASS] C[%0d][%0d]: HW Result = %6d | Golden Model = %6d", 
                             i, j, calculated_out, golden_C[i][j]);
                    passed_tests = passed_tests + 1;
                end else begin
                    $display("  [FAIL] C[%0d][%0d]: HW Result = %6d | Golden Model = %6d (MISMATCH)", 
                             i, j, calculated_out, golden_C[i][j]);
                    failed_tests = failed_tests + 1;
                end
            end
        end

        // =====================================================================
        // Test Suite 3: Deep Vector Accumulation & Dynamic Range
        // =====================================================================
        $display("\n[TEST 3] Testing High Dynamic Range Vector (64 elements)...");
        begin : test_deep_vector
            reg signed [31:0] expected_acc;
            expected_acc = 0;

            for (i = 0; i < 64; i = i + 1) begin
                @(posedge clk);
                stream_valid   <= 1'b1;
                stream_weight  <= 16'sd250 - i * 4;
                stream_act     <= 16'sd100 + i * 2;
                stream_clr_acc <= (i == 0) ? 1'b1 : 1'b0;
                stream_last    <= (i == 63) ? 1'b1 : 1'b0;
                expected_acc   = expected_acc + ((16'sd250 - i * 4) * (16'sd100 + i * 2));
            end

            @(posedge clk);
            stream_valid   <= 1'b0;
            stream_clr_acc <= 1'b0;
            stream_last    <= 1'b0;

            while (!out_valid) @(posedge clk);

            total_tests = total_tests + 1;
            if (out_data_32 === expected_acc) begin
                $display("  [PASS] 64-element Vector Accumulation: HW = %0d | Golden = %0d", out_data_32, expected_acc);
                passed_tests = passed_tests + 1;
            end else begin
                $display("  [FAIL] 64-element Vector Accumulation: HW = %0d | Golden = %0d", out_data_32, expected_acc);
                failed_tests = failed_tests + 1;
            end
        end

        // =====================================================================
        // Summary & Final Status Evaluation
        // =====================================================================
        $display("\n============================================================");
        $display("Verification Summary:");
        $display("  Total Tests Executed : %0d", total_tests);
        $display("  Tests Passed         : %0d", passed_tests);
        $display("  Tests Failed         : %0d", failed_tests);
        $display("============================================================");

        if (failed_tests == 0 && passed_tests > 0) begin
            $display("\n************************************************************");
            $display("  STATUS: SUCCESS");
            $display("  All RISC-V AI Acceleration MAC computations verified!");
            $display("************************************************************\n");
        end else begin
            $display("\n************************************************************");
            $display("  STATUS: FAILURE");
            $display("  Computation mismatches detected!");
            $display("************************************************************\n");
        end

        #(CLK_PERIOD * 10);
        $finish;
    end

endmodule
