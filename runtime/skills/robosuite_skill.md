# Robosuite Skill Library

Reusable workflow templates for robosuite-backed AgentOS tasks.

```json
{
  "schema_version": "embodied_lab.skill_library.v1",
  "skills": [
    {
      "name": "robosuite_lift",
      "description": "Lift the cube in robosuite Lift with a scripted 3D policy",
      "pattern": "lift the cube",
      "parameters": {},
      "steps": [
        {
          "tool": "reset_task",
          "parameters": {
            "instruction": "lift the cube",
            "target_color": "red",
            "receptacle_name": "bowl"
          },
          "description": "Reset the robosuite Lift task."
        },
        {
          "tool": "robosuite_lift_loop",
          "parameters": {
            "max_steps": 120,
            "render_every": 4,
            "frames_dir": "outputs/robosuite_lift_viz",
            "video_output": "outputs/robosuite_lift_viz/lift.gif",
            "visual_cameras": ["frontview"],
            "gif_duration_ms": 100,
            "continue_after_success_steps": 6
          },
          "description": "Run the robosuite scripted lift loop."
        },
        {
          "tool": "render_fake_env",
          "parameters": {},
          "description": "Render final robosuite state."
        }
      ]
    }
  ]
}
```
