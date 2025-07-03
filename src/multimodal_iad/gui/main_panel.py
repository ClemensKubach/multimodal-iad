import flet as ft


def main(page: ft.Page):
    page.title = "Multimodal-IAD"
    page.bgcolor = ft.colors.WHITE
    page.window_width = 900
    page.window_height = 700

    # Placeholder images (use a local file or a URL for real images)
    placeholder_img = "https://via.placeholder.com/300x200?text=Input+Image"
    placeholder_heatmap = "https://via.placeholder.com/300x200?text=Heatmap"

    # Input source selection
    input_source = ft.Dropdown(
        label="Input Source",
        options=[
            ft.dropdown.Option("Image File"),
            ft.dropdown.Option("Camera (RGB)"),
            ft.dropdown.Option("Camera (Depth)"),
        ],
        value="Image File",
        width=200,
    )

    # Image display
    input_image = ft.Image(src=placeholder_img, width=300, height=200, fit=ft.ImageFit.CONTAIN)
    heatmap_image = ft.Image(src=placeholder_heatmap, width=300, height=200, fit=ft.ImageFit.CONTAIN)

    # Results
    anomaly_label = ft.Text("Prediction: (placeholder)", size=20, color=ft.colors.BLACK, weight=ft.FontWeight.BOLD)
    anomaly_score = ft.Text("Score: (placeholder)", size=18, color=ft.colors.BLACK)
    explanation_text = ft.TextField(
        value="This is a placeholder explanation.",
        multiline=True,
        read_only=True,
        width=300,
        bgcolor=ft.colors.WHITE,
        color=ft.colors.BLACK,
        border_radius=8,
    )

    # Audio output (placeholder)
    def play_audio(e):
        page.snack_bar = ft.SnackBar(ft.Text("Audio output (placeholder)"))
        page.update()

    audio_btn = ft.ElevatedButton(
        "Play Audio Output",
        on_click=play_audio,
        bgcolor=ft.colors.BLUE_100,
        color=ft.colors.BLUE_900,
    )

    # Load image (placeholder)
    def load_image(e):
        page.dialog = ft.AlertDialog(
            title=ft.Text("Load Image"),
            content=ft.Text("Image loading not implemented in this demo."),
        )
        page.update()
        page.dialog.open = True
        page.update()

    load_btn = ft.ElevatedButton(
        "Load Image",
        on_click=load_image,
        bgcolor=ft.colors.BLUE_100,
        color=ft.colors.BLUE_900,
    )

    # Layout
    page.add(
        ft.Column(
            [
                ft.Row(
                    [
                        input_source,
                        load_btn,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                ),
                ft.Row(
                    [
                        ft.Column([input_image, heatmap_image], spacing=20),
                        ft.Column(
                            [
                                anomaly_label,
                                anomaly_score,
                                ft.Text("Textual Explanation:", size=16, color=ft.colors.BLUE_GREY_700),
                                explanation_text,
                                audio_btn,
                            ],
                            spacing=16,
                        ),
                    ],
                    spacing=40,
                ),
            ],
            spacing=30,
        ),
    )
