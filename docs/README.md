# VAROMATIC+ Presentation

This directory contains the presentation files for the VAROMATIC+ football analysis application.

## Structure

- `presentation.html` - The main presentation file
- `assets/` - Directory containing images and other media files
- `VAROMATIC+_Presentation.md` - Markdown version of the presentation
- `create_pptx.py` - Script to convert the presentation to PowerPoint format (requires python-pptx)
- `create_simple_pptx.py` - Simplified version of the PowerPoint conversion script

## Using the Presentation

### Web Version (Recommended)

1. Open `presentation.html` in a modern web browser
2. Use arrow keys (← →) to navigate between slides
3. Press 'C' when hovering over feature cards to view code examples
4. Click on code tabs to switch between different code sections
5. Use the copy button to copy code snippets
6. ESC key closes code overlays

### Navigation Controls

- **Keyboard**:
  - Left Arrow (←) - Previous slide
  - Right Arrow (→) - Next slide
  - 'C' key - View code when hovering over feature cards
  - ESC - Close code overlay

- **Touch Devices**:
  - Swipe left/right to navigate between slides
  - Tap feature cards to view code
  - Tap close button to exit code view

### Features

- Responsive design that works on all devices
- Smooth transitions and animations
- Code syntax highlighting
- Interactive code demonstrations
- Progress indicator
- Touch-friendly interface

## Technical Requirements

The presentation uses the following external libraries (loaded via CDN):

- Font Awesome 6.0.0
- AOS (Animate On Scroll) 2.3.1
- Prism.js 1.24.1 (for code syntax highlighting)

No local installation is required as all dependencies are loaded from CDN.

## Troubleshooting

If you experience any issues:

1. Ensure you're using a modern web browser (Chrome, Firefox, Safari, Edge)
2. Check that JavaScript is enabled
3. Verify that you have an internet connection (for loading CDN resources)
4. Clear browser cache if styles or scripts aren't loading properly

## Converting to PowerPoint

While the web version is recommended for the best experience, you can convert the presentation to PowerPoint format:

1. Install required Python packages:
   ```bash
   pip install python-pptx Pillow
   ```

2. Run the conversion script:
   ```bash
   python create_pptx.py
   ```

Note: The PowerPoint version may not include all interactive features available in the web version.

## Assets
The `assets` directory contains:
- banner.png: Title slide banner
- ui_screenshot.png: VAROMATIC+ interface screenshot
- thank_you.png: Final slide image

## Customization
Feel free to:
- Modify the color scheme
- Add your own images
- Update contact information
- Customize transitions
- Add your organization's branding 