# 🚀 Deployment Guide for Hugging Face Spaces

This guide will help you deploy your Hugging Face Pipelines demo to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: Create a free account at [huggingface.co](https://huggingface.co)
2. **Git**: Make sure you have Git installed on your machine
3. **Hugging Face CLI**: Install the Hugging Face CLI tools

## Step 1: Install Hugging Face CLI

```bash
pip install huggingface_hub[cli]
```

## Step 2: Login to Hugging Face

```bash
huggingface-cli login
```

Enter your Hugging Face token when prompted.

## Step 3: Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Space name**: `hf-pipelines-demo` (or your preferred name)
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free tier)
   - **Visibility**: Public

## Step 4: Clone and Setup

```bash
# Clone your space (replace with your username and space name)
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy the files from this project
cp /path/to/your/hf_pipeline/app.py .
cp /path/to/your/hf_pipeline/requirements_spaces.txt requirements.txt
cp /path/to/your/hf_pipeline/README_spaces.md README.md
```

## Step 5: Deploy

```bash
# Add and commit files
git add .
git commit -m "Initial commit: Hugging Face Pipelines Demo"

# Push to deploy
git push
```

## Step 6: Monitor Deployment

1. Go to your Space page on Hugging Face
2. Check the "Logs" tab for any errors
3. The app should be available at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## File Structure for Spaces

Your Space should have this structure:
```
your-space/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # Space description and metadata
└── .gitignore           # Optional: ignore unnecessary files
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: 
   - The app is optimized for CPU and includes model caching
   - If you get memory errors, try reducing the model sizes in `app.py`

2. **Slow Loading**:
   - First-time model downloads can take 5-10 minutes
   - Subsequent runs will be faster due to caching

3. **Model Loading Errors**:
   - Some models might fail to load on the free tier
   - Check the logs for specific error messages

### Performance Optimization

- **Model Caching**: Models are cached after first load
- **Lazy Loading**: Models are only loaded when needed
- **CPU Optimization**: All models use CPU-optimized settings

## Customization

### Adding New Pipelines

To add new pipelines to the app:

1. Create a new function in `app.py`
2. Add a new tab in the Gradio interface
3. Connect the function to the interface

### Changing Models

To use different models:

1. Modify the `get_pipeline()` function calls
2. Update the model names in the pipeline initialization
3. Test locally before deploying

## Monitoring and Maintenance

- **Logs**: Check the Space logs regularly for errors
- **Usage**: Monitor usage in the Space dashboard
- **Updates**: Keep dependencies updated for security

## Cost Considerations

- **Free Tier**: Includes CPU Basic hardware (2 vCPU, 16GB RAM)
- **Paid Tiers**: Available for higher performance needs
- **Storage**: Models are cached, so storage usage is minimal

## Support

If you encounter issues:
1. Check the Hugging Face Spaces documentation
2. Review the logs in your Space
3. Test locally first to isolate issues
4. Consider the free tier limitations

## Next Steps

After successful deployment:
1. Share your Space with others
2. Add more advanced features
3. Consider upgrading to paid tiers for better performance
4. Explore other Hugging Face Spaces for inspiration
