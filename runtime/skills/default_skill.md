# Skill Library

Reusable workflow templates for AgentOS.

```json
{
  "schema_version": "embodied_lab.skill_library.v1",
  "skills": [
    {
      "name": "pick_place",
      "description": "Pick up a colored block and place it in the bowl",
      "pattern": "pick up the {color} block and place it in the bowl",
      "parameters": {
        "color": {
          "type": "string",
          "values": ["red", "blue", "green"]
        }
      },
      "target_color_param": "color",
      "steps": [
        {
          "tool": "reset_task",
          "parameters": {
            "instruction": "pick up the {color} block and place it in the bowl",
            "target_color": "{color}",
            "receptacle_name": "bowl"
          },
          "description": "Initialize the task."
        },
        {
          "tool": "scripted_pick_place_loop",
          "parameters": {"max_steps": 80},
          "description": "Run the expert loop."
        },
        {
          "tool": "render_fake_env",
          "parameters": {},
          "description": "Render final state."
        }
      ]
    }
  ]
}
```
