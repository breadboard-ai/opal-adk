from unittest import mock

from absl.testing import absltest
from google.genai import types

from opal_adk.tools.generate import generate_images
from opal_adk.tools.generate.generate_utils import gemini_generate_image
from opal_adk.types import image_types


class GenerateImagesTest(absltest.TestCase):

  @mock.patch.object(
      gemini_generate_image,
      'gemini_generate_images',
      autospec=True,
  )
  def test_generate_images_pro(self, mock_generate_image):
    mock_generate_image.return_value = [
        (b'image1', 'image/png'),
        (b'image2', 'image/jpeg'),
    ]

    images_to_edit = [
        types.Content(
            parts=[
                types.Part.from_bytes(data=b'input1', mime_type='image/png')
            ]
        ),
        types.Content(
            parts=[
                types.Part.from_bytes(data=b'input2', mime_type='image/png')
            ]
        ),
    ]

    result = generate_images.generate_images(
        prompt='A cool cat',
        model='pro',
        images=images_to_edit,
        aspect_ratio='16:9',
    )

    self.assertLen(result, 2)
    self.assertEqual(
        result[0].parts[0].inline_data.data, b'image1'
    )
    self.assertEqual(
        result[0].parts[0].inline_data.mime_type, 'image/png'
    )
    self.assertEqual(
        result[1].parts[0].inline_data.data, b'image2'
    )
    self.assertEqual(
        result[1].parts[0].inline_data.mime_type, 'image/jpeg'
    )

    mock_generate_image.assert_called_once()
    _, kwargs = mock_generate_image.call_args
    self.assertEqual(kwargs['model_name'], 'gemini-3-pro-image-opal')
    self.assertEqual(kwargs['aspect_ratio'], image_types.AspectRatio('16:9'))
    self.assertLen(kwargs['parts'], 3)
    self.assertIn('A cool cat', kwargs['parts'][0].text)

  @mock.patch.object(
      gemini_generate_image,
      'gemini_generate_images',
      autospec=True,
  )
  def test_generate_images_flash(self, mock_generate_image):
    mock_generate_image.return_value = [
        (b'image1', 'image/png'),
    ]

    result = generate_images.generate_images(
        prompt='A cool dog',
        model='flash',
        images=[],
        aspect_ratio='1:1',
    )

    self.assertLen(result, 1)
    mock_generate_image.assert_called_once()
    _, kwargs = mock_generate_image.call_args
    self.assertEqual(kwargs['model_name'], 'gemini-2.5-flash-image')
    self.assertEqual(kwargs['aspect_ratio'], image_types.AspectRatio('1:1'))


if __name__ == '__main__':
  absltest.main()
