// TypeScript types for Story Scheduling API

export interface StoryInput {
  story: string;
  title: string;
  writer: string;
}

export interface SchedulingRequest {
  day_type: "weekday" | "weekend";
  stories: StoryInput[];
}

export interface ScheduledSlot {
  slot: number;
  story: string;
  title: string;
  writer: string;
  writer_tier: string;
  recommended_post_time: string;
  recommended_day: string;
  daypart: string;
  estimated_read_length: string;
  content_category: string;
  emotional_tone: string;
  scheduling_rules_applied: number[];
  golden_slot_used: boolean;
  reasoning: string;
  flags: string | null;
}

export interface SchedulingResponse {
  success: boolean;
  message: string;
  schedule?: ScheduledSlot[];
  total_stories?: number;
  golden_slots_used?: number;
  s_tier_count?: number;
}

