const std = @import("std");
const vk = @import("vulkan");

const H264MbSizeAlignment = 16;

fn alignSize(comptime T: type, size: T, alignment: T) T {
    std.debug.assert((alignment & (alignment - 1)) == 0); // Ensure power of two
    return (size + alignment - 1) & ~(@as(T, alignment - 1));
}

pub fn getStdVideoH264SequenceParameterSetVui(fps: u32) vk.StdVideoH264SequenceParameterSetVui {
    const flags = vk.StdVideoH264SpsVuiFlags{
        .timing_info_present_flag = true,
        .fixed_frame_rate_flag = true,
    };

    var ret = std.mem.zeroes(vk.StdVideoH264SequenceParameterSetVui);
    ret.flags = flags;
    ret.num_units_in_tick = 1;
    ret.time_scale = fps * 2;

    return ret;
}

pub fn getStdVideoH264SequenceParameterSet(
    width: u32,
    height: u32,
    p_vui: ?*const vk.StdVideoH264SequenceParameterSetVui,
) vk.StdVideoH264SequenceParameterSet {
    const flags = vk.StdVideoH264SpsFlags{
        .direct_8x_8_inference_flag = true,
        .frame_mbs_only_flag = true,
        .vui_parameters_present_flag = true,
    };

    const mb_aligned_width = alignSize(u32, width, H264MbSizeAlignment);
    const mb_aligned_height = alignSize(u32, height, H264MbSizeAlignment);

    var ret = std.mem.zeroes(vk.StdVideoH264SequenceParameterSet);
    ret.profile_idc = .main;
    ret.level_idc = .@"4_1";
    ret.seq_parameter_set_id = 0;
    ret.chroma_format_idc = .@"420";
    ret.bit_depth_luma_minus_8 = 0;
    ret.bit_depth_chroma_minus_8 = 0;
    ret.log_2_max_frame_num_minus_4 = 0;
    ret.pic_order_cnt_type = .@"0";
    ret.max_num_ref_frames = 1;
    ret.pic_width_in_mbs_minus_1 = mb_aligned_width / H264MbSizeAlignment - 1;
    ret.pic_height_in_map_units_minus_1 = mb_aligned_height / H264MbSizeAlignment - 1;
    ret.flags = flags;
    ret.p_sequence_parameter_set_vui = p_vui.?;
    ret.frame_crop_right_offset = mb_aligned_width - width;
    ret.frame_crop_bottom_offset = mb_aligned_height - height;

    // This allows for picture order count values in the range [0, 255].
    ret.log_2_max_pic_order_cnt_lsb_minus_4 = 4;

    if (ret.frame_crop_right_offset > 0 or ret.frame_crop_bottom_offset > 0) {
        ret.flags.frame_cropping_flag = true;

        if (ret.chroma_format_idc == .@"420") {
            ret.frame_crop_right_offset >>= 1;
            ret.frame_crop_bottom_offset >>= 1;
        }
    }

    return ret;
}

pub fn getStdVideoH264PictureParameterSet() vk.StdVideoH264PictureParameterSet {
    const flags = vk.StdVideoH264PpsFlags{
        //.transform_8x8_mode_flag = 1;
        .deblocking_filter_control_present_flag = true,
        .entropy_coding_mode_flag = true,
    };

    var pps = std.mem.zeroes(vk.StdVideoH264PictureParameterSet);
    pps.flags = flags;

    return pps;
}

pub const FrameInfo = struct {
    const Self = @This();

    slice_header_flags: vk.StdVideoEncodeH264SliceHeaderFlags = std.mem.zeroes(vk.StdVideoEncodeH264SliceHeaderFlags),
    slice_header: vk.StdVideoEncodeH264SliceHeader = std.mem.zeroes(vk.StdVideoEncodeH264SliceHeader),
    slice_info: vk.VideoEncodeH264NaluSliceInfoKHR = std.mem.zeroes(vk.VideoEncodeH264NaluSliceInfoKHR),
    picture_info_flags: vk.StdVideoEncodeH264PictureInfoFlags = std.mem.zeroes(vk.StdVideoEncodeH264PictureInfoFlags),
    std_picture_info: vk.StdVideoEncodeH264PictureInfo = std.mem.zeroes(vk.StdVideoEncodeH264PictureInfo),
    encode_h264_frame_info: vk.VideoEncodeH264PictureInfoKHR = std.mem.zeroes(vk.VideoEncodeH264PictureInfoKHR),
    reference_lists: vk.StdVideoEncodeH264ReferenceListsInfo = std.mem.zeroes(vk.StdVideoEncodeH264ReferenceListsInfo),

    pub fn init(
        self: *Self,
        frame_count: u32,
        width: u32,
        height: u32,
        sps: vk.StdVideoH264SequenceParameterSet,
        pps: vk.StdVideoH264PictureParameterSet,
        gop_frame_count: u32,
        use_constant_qp: bool,
    ) void {
        _ = width;
        _ = height;
        const is_i = gop_frame_count == 0;
        // TODO: check this u5 cast
        const max_pic_order_cnt_lsb = @as(u32, 1) << @as(u5, @intCast((sps.log_2_max_pic_order_cnt_lsb_minus_4 + @as(u32, 4))));

        self.slice_header_flags.direct_spatial_mv_pred_flag = true;
        self.slice_header_flags.num_ref_idx_active_override_flag = false;

        self.slice_header.flags = self.slice_header_flags;
        self.slice_header.slice_type = if (is_i) .i else .p;
        self.slice_header.cabac_init_idc = .@"0";
        self.slice_header.disable_deblocking_filter_idc = .disabled;
        self.slice_header.slice_alpha_c_0_offset_div_2 = 0;
        self.slice_header.slice_beta_offset_div_2 = 0;

        const picWidthInMbs = sps.pic_width_in_mbs_minus_1 + 1;
        const picHeightInMbs = sps.pic_height_in_map_units_minus_1 + 1;
        _ = picWidthInMbs * picHeightInMbs; // Unused but included for reference

        self.slice_info.s_type = .video_encode_h264_nalu_slice_info_khr;
        self.slice_info.p_next = null;
        self.slice_info.p_std_slice_header = &self.slice_header;
        self.slice_info.constant_qp = if (use_constant_qp) pps.pic_init_qp_minus_26 + 26 else 0;

        self.picture_info_flags.idr_pic_flag = is_i;
        self.picture_info_flags.is_reference = true;
        self.picture_info_flags.adaptive_ref_pic_marking_mode_flag = false;
        self.picture_info_flags.no_output_of_prior_pics_flag = is_i;

        self.std_picture_info.flags = self.picture_info_flags;
        self.std_picture_info.seq_parameter_set_id = 0;
        self.std_picture_info.pic_parameter_set_id = pps.pic_parameter_set_id;
        self.std_picture_info.idr_pic_id = 0;
        self.std_picture_info.primary_pic_type = if (is_i) .idr else .p;
        // frame_num is incremented for each reference frame transmitted.
        // In our case, only the first frame (which is IDR) is a reference
        // frame with frame_num == 0, and all others have frame_num == 1.
        self.std_picture_info.frame_num = frame_count;

        // POC is incremented by 2 for each coded frame.
        self.std_picture_info.pic_order_cnt = @as(i32, @intCast((frame_count * 2) % max_pic_order_cnt_lsb));
        self.reference_lists.num_ref_idx_l_0_active_minus_1 = 0;
        self.reference_lists.num_ref_idx_l_1_active_minus_1 = 0;
        // TODO: double check these memsets
        @memset(&self.reference_lists.ref_pic_list_0, vk.STD_VIDEO_H264_NO_REFERENCE_PICTURE);
        @memset(&self.reference_lists.ref_pic_list_1, vk.STD_VIDEO_H264_NO_REFERENCE_PICTURE);
        if (!is_i) {
            self.reference_lists.ref_pic_list_0[0] = @intFromBool((gop_frame_count & 1) == 0);
        }
        std.debug.print("ref: {any}\n", .{self.reference_lists.ref_pic_list_0});
        self.std_picture_info.p_ref_lists = &self.reference_lists;

        self.encode_h264_frame_info.s_type = .video_encode_h264_picture_info_khr;
        self.encode_h264_frame_info.p_next = null;
        self.encode_h264_frame_info.nalu_slice_entry_count = 1;
        self.encode_h264_frame_info.p_nalu_slice_entries = @ptrCast(&self.slice_info);
        self.encode_h264_frame_info.p_std_picture_info = &self.std_picture_info;
    }

    pub fn getEncodeH264FrameInfo(self: *const FrameInfo) *const vk.VideoEncodeH264PictureInfoKHR {
        return &self.encode_h264_frame_info;
    }
};
